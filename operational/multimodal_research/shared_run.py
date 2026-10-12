"""Scoped capacity admission for two small head jobs; never a latency benchmark."""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from unittest.mock import patch

import psutil
import torch

from operational.multimodal_validation.common import read, sha, write

from . import common

ALLOWED_PREFIX = "operational.representation_audit."
ACTIONS = {
    "train": ("train", ["--mode", "both"]),
    "cache_v": ("cache", ["V"]),
    "evaluate": ("evaluate", []),
    "units": ("phase_units", ["train"]),
    "evaluate_units": ("evaluate", ["--study", "units"]),
    "parity": ("route_parity", []),
    "next_studies": ("next_studies", ["train"]),
    "evaluate_units_s13": ("evaluate_next", ["UNIT_REPLICATION", "13"]),
    "evaluate_units_s23": ("evaluate_next", ["UNIT_REPLICATION", "23"]),
    "evaluate_relative": ("evaluate_next", ["RELATIVE_PHASE", "7"]),
}


def nvidia_query(field: str) -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", field, "--format=csv,noheader,nounits"], text=True, timeout=10
        )
    except subprocess.SubprocessError as error:
        raise RuntimeError("GPU resource observation failed") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=tuple(ACTIONS))
    args = parser.parse_args()
    policy: dict = dict(
        source_sha256=sha(__file__),
        allowed_foreign_module_prefix=ALLOWED_PREFIX,
        owner_detection="Only live NVIDIA compute PIDs; CPU preparation is not a GPU owner",
        free_memory="Minimum of CUDA and nvidia-smi free memory",
        max_foreign_head_jobs=1,
        min_free_cuda_bytes=6 * 1024**3,
        max_own_reserved_cuda_bytes=1024**3,
        unchanged="Native budget, elapsed deadline, RAM/disk admission, training recipe and RNG",
        reason="Exclusive admission starved the small readout while successive V17 head queues ran",
        scope="Head fitting, frozen-state extraction and accuracy only; no shared latency claims",
        V15_closed=common.authorize()["V15_closeout"],
        scientific_sources=common.source_hashes(),
    )
    path = common.OUT / "admission" / ("SHARED_HEAD_" + sha(__file__) + ".json")
    if path.exists():
        assert read(path) == policy
    else:
        write(path, policy)
    assert read(common.OUT / "cache/H/BINDING.json")["sources"] == common.source_hashes()
    original = common.guard
    receipt_path = common.OUT / f"SHARED_EXECUTION_{args.action}_{os.getpid()}.json"
    receipt = dict(
        action=args.action,
        pid=os.getpid(),
        utc_start=datetime.now(UTC).isoformat(),
        policy_sha256=sha(path),
        accounting_before=common.accounting(),
        status="RUNNING",
    )
    write(receipt_path, receipt)

    def admitted(*, prospective_updates: int = 0) -> None:
        # Native time/update/RAM/disk checks execute before its exclusive-owner check.
        try:
            original(prospective_updates=prospective_updates)
        except RuntimeError as error:
            if not str(error).startswith("Live research GPU owner must finish:"):
                raise
            gpu_pids = {
                int(line.strip())
                for line in nvidia_query("--query-compute-apps=pid").splitlines()
                if line.strip().isdigit()
            }
            own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
            foreign = []
            for process in psutil.process_iter(["name", "cmdline", "ppid"]):
                try:
                    if (
                        process.pid in own
                        or process.pid not in gpu_pids
                        or not (process.info["name"] or "").lower().startswith("python")
                    ):
                        continue
                    command = process.info["cmdline"] or []
                    if "-m" not in command:
                        continue
                    module = command[command.index("-m") + 1]
                    if not module.startswith("operational."):
                        continue
                    if not module.startswith(ALLOWED_PREFIX):
                        raise RuntimeError(
                            "Shared admission rejects this live module: " + module
                        ) from error
                    foreign.append(dict(pid=process.pid, ppid=process.info["ppid"], module=module))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            pids = {p["pid"] for p in foreign}
            roots = [p for p in foreign if p["ppid"] not in pids]
            free, total = torch.cuda.mem_get_info()
            nvidia_free = (
                int(nvidia_query("--query-gpu=memory.free").splitlines()[0].strip())
                * 1024**2
            )
            free = min(free, nvidia_free)
            reserved = torch.cuda.memory_reserved()
            if (
                len(roots) > 1
                or free < policy["min_free_cuda_bytes"]
                or reserved > policy["max_own_reserved_cuda_bytes"]
            ):
                raise RuntimeError("Shared CUDA capacity unavailable") from error
            write(
                common.OUT / "SHARED_ADMISSION_LATEST.json",
                dict(
                    utc=datetime.now(UTC).isoformat(),
                    own_pid=os.getpid(),
                    foreign=foreign,
                    free_cuda_bytes=free,
                    total_cuda_bytes=total,
                    own_reserved_cuda_bytes=reserved,
                    policy_sha256=sha(path),
                    prospective_updates=prospective_updates,
                ),
            )

    module, arguments = ACTIONS[args.action]
    sys.argv = [module, *arguments]
    if args.action == "parity":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.set_per_process_memory_fraction(
            policy["max_own_reserved_cuda_bytes"] / torch.cuda.get_device_properties(0).total_memory
        )
    try:
        with patch.object(common, "guard", admitted):
            runpy.run_module("operational.multimodal_research." + module, run_name="__main__")
        receipt["status"] = "COMPLETE"
    except BaseException as error:
        receipt["status"] = "STOPPED"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["utc_end"] = datetime.now(UTC).isoformat()
        receipt["accounting_after"] = common.accounting()
        write(receipt_path, receipt)


if __name__ == "__main__":
    main()
