"""Record detector prerequisites without treating them as a detector/G1 pass.

This standalone, standard-library probe never installs packages, compiles code,
downloads weights, or trains a model. Exit 2 means G1 remains blocked. CUDA
primitive measurements, if requested, are explicitly not model measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PINNED_SOURCES = {
    "monodgp": "aa059a18214aebf644510e7f0793971b403f9d14",
    "monodetr": "6994b9f512400b258c6edb75f77423beb9c126f2",
}

TORCH_PROBE = r"""
import importlib, json
result = {"import": "NOT_RUN", "cuda_available": None, "devices": [],
          "primitive_status": "NOT_RUN", "extension_import": "NOT_RUN"}
try:
    import torch
    result.update({"import": "PASS", "version": torch.__version__,
                   "cuda_build": torch.version.cuda,
                   "cuda_available": bool(torch.cuda.is_available()),
                   "compiled_architectures": torch.cuda.get_arch_list()})
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        result["devices"].append({"index": i, "name": p.name,
          "capability": list(torch.cuda.get_device_capability(i)),
          "total_memory_bytes": p.total_memory})
    if result["cuda_available"] and RUN_PRIMITIVES:
        torch.manual_seed(7)
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(128, 128, device="cuda", requires_grad=True)
        loss = (x @ x.T).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(loss).item() and
                      x.grad is not None and torch.isfinite(x.grad).all().item())
        result["primitive_status"] = "PASS" if finite else "FAIL"
        result["primitive_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["primitive_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["primitive_scope"] = "matrix_product_only_not_detector"
    if result["cuda_available"]:
        try:
            extension = importlib.import_module("MultiScaleDeformableAttention")
            result["extension_import"] = "PASS"
            result["extension_file"] = extension.__file__
        except Exception as exc:
            result["extension_import"] = "FAIL"
            result["extension_error"] = type(exc).__name__ + ": " + str(exc)
except Exception as exc:
    result["error"] = type(exc).__name__ + ": " + str(exc)
    if result["import"] != "PASS":
        result["import"] = "FAIL"
    else:
        result["primitive_status"] = "FAIL"
print(json.dumps(result, allow_nan=False))
"""


def command_record(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    """Capture a bounded command, including missing-command and timeout failures."""
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "error": str(exc)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def source_record(name: str, root: Path) -> dict[str, Any]:
    """Hash inspected upstream files; do not import or modify their code."""
    head = command_record(["git", "rev-parse", "HEAD"], cwd=root)
    diff = command_record(["git", "diff", "HEAD", "--"], cwd=root)
    expected = PINNED_SOURCES[name]
    actual = head.get("stdout") if head.get("returncode") == 0 else None
    files = [
        "README.md",
        "requirements.txt",
        "LICENSE" if name == "monodgp" else "LICENSE.txt",
        f"configs/{name}.yaml",
        f"lib/models/{name}/ops/setup.py",
        f"lib/models/{name}/ops/test.py",
        f"lib/models/{name}/ops/src/cuda/ms_deform_attn_cuda.cu",
        f"lib/models/{name}/ops/src/cuda/ms_deform_im2col_cuda.cuh",
        "lib/datasets/kitti/kitti_dataset.py",
        "lib/helpers/decode_helper.py",
    ]
    hashes = {}
    for relative in files:
        path = root / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {
        "root": str(root.resolve()),
        "expected_revision": expected,
        "observed_revision": actual,
        "revision_matches": actual == expected,
        "tracked_changes_present": (bool(diff["stdout"]) if diff.get("returncode") == 0 else None),
        "tracked_diff_sha256": (
            hashlib.sha256(diff["stdout"].encode()).hexdigest()
            if diff.get("returncode") == 0
            else None
        ),
        "files_sha256": hashes,
    }


def collect_report(
    candidate_roots: dict[str, Path] | None = None, *, run_cuda_primitives: bool = False
) -> dict[str, Any]:
    """Return provenance, prerequisite observations, and explicitly unrun G1 checks."""
    packages = {}
    for name in ("torch", "torchvision", "numpy", "pytest", "ruff"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    probe = command_record(
        [sys.executable, "-I", "-c", f"RUN_PRIMITIVES = {run_cuda_primitives!r}\n" + TORCH_PROBE]
    )
    torch: dict[str, Any] = {"import": "FAIL", "error": "probe did not return valid JSON"}
    if probe.get("returncode") == 0:
        try:
            decoded = json.loads(probe["stdout"])
            if isinstance(decoded, dict):
                torch = decoded
        except (json.JSONDecodeError, TypeError):
            pass
    blockers = []
    if torch.get("import") != "PASS":
        blockers.append("torch_import_unavailable")
    if torch.get("cuda_available") is False:
        blockers.append("cuda_device_unavailable")
    elif torch.get("cuda_available") is not True:
        blockers.append("cuda_device_access_unverified")
    if shutil.which("nvcc") is None:
        blockers.append("cuda_compiler_not_on_path")
    if torch.get("extension_import") != "PASS":
        blockers.append("deformable_extension_not_imported")
    blockers.append("real_detector_forward_backward_decode_not_executed_by_this_probe")
    candidates = {name: source_record(name, path) for name, path in (candidate_roots or {}).items()}
    for name, source in candidates.items():
        if not source["revision_matches"]:
            blockers.append(f"{name}_source_revision_mismatch")
    return {
        "schema_version": 1,
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "prerequisite_inventory_not_detector_validation",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "linux_nvidia_device_nodes": sorted(str(path) for path in Path("/dev").glob("nvidia*")),
        "isolated_environment": sys.prefix != sys.base_prefix,
        "packages": packages,
        "executables": {name: shutil.which(name) for name in ("nvidia-smi", "nvcc", "c++", "git")},
        "nvidia_smi": command_record(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "nvcc": command_record(["nvcc", "--version"]),
        "torch": torch,
        "torch_probe_process": {key: value for key, value in probe.items() if key != "command"},
        "candidate_sources": candidates,
        "g1": {
            "status": "BLOCKED",
            "blockers": blockers,
            "detector_forward": "NOT_RUN",
            "detector_loss_and_backward": "NOT_RUN",
            "detector_decode": "NOT_RUN",
            "operator_numerical_parity": "NOT_RUN",
            "detector_peak_memory_bytes": None,
        },
    }


def main() -> int:
    """Write a machine-readable report and exit nonzero while G1 is blocked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monodgp-root", type=Path)
    parser.add_argument("--monodetr-root", type=Path)
    parser.add_argument("--run-cuda-primitives", action="store_true")
    args = parser.parse_args()
    roots = {
        name: value
        for name, value in {"monodgp": args.monodgp_root, "monodetr": args.monodetr_root}.items()
        if value is not None
    }
    report = collect_report(roots, run_cuda_primitives=args.run_cuda_primitives)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"g1": report["g1"], "report": str(args.output)}, allow_nan=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
