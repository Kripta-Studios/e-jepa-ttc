"""Resume native GPU replay with the already-audited exact CPU input preparation."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from operational.fcwd_rgb_registration import transfer as fcwd_engine
from operational.multimodal_validation.common import MANIFEST, read, sha, write

from . import native_gpu_run
from .common import OUT
from .exact_event_route import compact_union


def inventory() -> dict:
    directory = OUT / "external_states_gpu_native/FCWD"
    return {path.name: sha(path) for path in directory.glob("*.json") if len(path.stem) == 64}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "run"))
    args = parser.parse_args()
    source = Path(__file__).parent
    binding = dict(
        source_sha256=sha(__file__),
        adapter_sources={
            name: sha(source / name)
            for name in (
                "exact_event_route.py",
                "exact_windows.py",
                "selected_map.py",
            )
        },
        existing_evidence_sha256=sha(OUT / "EXACT_EVENT_ROUTE.json"),
        manifest_sha256=sha(MANIFEST),
        change="Exact CPU window/map reuse only; GPU encoder batches and all models unchanged",
        resource_guards="unchanged budget, elapsed deadline, RAM/disk and GPU admission",
    )
    verification_path = OUT / "NATIVE_GPU_EXACT_PREPARATION.json"
    if args.action == "verify":
        torch.set_num_threads(1)
        rows = {row["query_id"]: row for row in read(MANIFEST)["rows"]}
        checks = []
        for qid in ("FCWD2:0145", "FCWD2:0146"):
            row = rows[qid]
            baseline = fcwd_engine.prepare_event(row)["own_events"]
            with patch.object(fcwd_engine, "encode_union", compact_union(fcwd_engine.encode_union)):
                optimized = fcwd_engine.prepare_event(row)["own_events"]
            assert baseline.dtype == optimized.dtype and np.array_equal(baseline, optimized), qid
            checks.append(dict(query_id=qid, exact=True, shape=list(baseline.shape)))
        write(
            verification_path,
            dict(
                binding=binding,
                checks=checks,
                selection="memory-pause boundary; no TTC consulted",
                optimizer_updates=0,
                targets_read=False,
            ),
        )
        print(checks, flush=True)
        return
    assert read(verification_path)["binding"] == binding
    before = inventory()
    receipt_path = OUT / f"EXACT_NATIVE_EXECUTION_{os.getpid()}.json"
    receipt: dict = dict(
        utc_start=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        verification_sha256=sha(verification_path),
        before_count=len(before),
        status="RUNNING",
    )
    write(receipt_path, receipt)
    try:
        with (
            patch.object(fcwd_engine, "encode_union", compact_union(fcwd_engine.encode_union)),
            patch.object(sys, "argv", ["native_gpu_run", "FCWD"]),
        ):
            native_gpu_run.main()
        receipt["status"] = "COMPLETE"
    except BaseException as error:
        receipt["status"] = "STOPPED"
        receipt["error"] = str(error)
        raise
    finally:
        after = inventory()
        assert all(after[key] == value for key, value in before.items())
        receipt["new_query_receipts"] = {
            key: value for key, value in after.items() if key not in before
        }
        receipt["utc_end"] = datetime.now(UTC).isoformat()
        write(receipt_path, receipt)


if __name__ == "__main__":
    main()
