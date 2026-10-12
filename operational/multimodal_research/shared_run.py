"""Scoped capacity admission for two small head jobs; never a latency benchmark."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from datetime import UTC, datetime
from unittest.mock import patch

import psutil
import torch

from operational.multimodal_validation.common import read, sha, write

from . import common

ALLOWED = {
    "operational.representation_audit.run",
    "operational.representation_audit.precision",
    "operational.representation_audit.directional",
}
CPU_ONLY = {
    "operational.multimodal_research.external_states",
    "operational.multimodal_research.external_evaluate",
    "operational.multimodal_research.head_attribution",
    "operational.multimodal_research.input_contributions",
}
ACTIONS = {
    "train": ("train", ["--mode", "both"]),
    "cache_v": ("cache", ["V"]),
    "evaluate": ("evaluate", []),
    "units": ("phase_units", ["train"]),
    "evaluate_units": ("evaluate", ["--study", "units"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=tuple(ACTIONS))
    args = parser.parse_args()
    policy: dict = dict(
        source_sha256=sha(__file__),
        allowed_foreign_modules=sorted(ALLOWED),
        max_foreign_head_jobs=1,
        min_free_cuda_bytes=6 * 1024**3,
        max_own_reserved_cuda_bytes=1024**3,
        unchanged="Native budget, elapsed deadline, RAM/disk admission, training recipe and RNG",
        reason="Exclusive admission starved the small readout while successive V17 head queues ran",
        scope="Head fitting, frozen-state extraction and accuracy only; no shared latency claims",
        V15_closed=common.authorize()["V15_closeout"],
        scientific_sources=common.source_hashes(),
    )
    path = common.OUT / "SHARED_HEAD_ADMISSION.json"
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
            own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
            foreign = []
            for process in psutil.process_iter(["name", "cmdline", "ppid"]):
                try:
                    if process.pid in own or not (process.info["name"] or "").lower().startswith(
                        "python"
                    ):
                        continue
                    command = process.info["cmdline"] or []
                    if "-m" not in command:
                        continue
                    module = command[command.index("-m") + 1]
                    if not module.startswith("operational.") or module in CPU_ONLY:
                        continue
                    if module not in ALLOWED:
                        raise RuntimeError(
                            "Shared admission rejects this live module: " + module
                        ) from error
                    foreign.append(dict(pid=process.pid, ppid=process.info["ppid"], module=module))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            pids = {p["pid"] for p in foreign}
            roots = [p for p in foreign if p["ppid"] not in pids]
            free, total = torch.cuda.mem_get_info()
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
