"""Resume Dev32 fusion replay with audited exact CPU input preparation."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from operational.multimodal_validation.common import V12, read, sha, write
from operational.rgb_port import transfer as engine
from operational.rgb_port_revision import transfer_inputs as dev_inputs

from . import dev32_fusion_states
from .common import OUT, authorize
from .exact_event_route import compact_union

MANIFEST = V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json"
NATIVE_ADMISSION = dev32_fusion_states.admission


def bounded_admission() -> dict:
    authorization = authorize()
    elapsed = (datetime.now(UTC) - datetime.fromisoformat(authorization["utc"])).total_seconds()
    if elapsed >= authorization["max_elapsed_seconds"]:
        raise RuntimeError("Authorized 18-hour execution window exhausted")
    return NATIVE_ADMISSION()


def inventory() -> dict:
    directory = OUT / "external_fusion_states/DEV32"
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
        change="Exact CPU window/map reuse only; encoder batches and all models unchanged",
        resource_guards="native RAM/disk and per-query elapsed deadline; CPU-only, zero updates",
    )
    verification_path = OUT / "DEV32_EXACT_PREPARATION.json"
    if args.action == "verify":
        torch.set_num_threads(1)
        rows = {row["query_id"]: row for row in read(MANIFEST)["rows"]}
        checks = []
        for qid in [row["query_id"] for row in read(MANIFEST)["rows"][73:75]]:
            row = rows[qid]
            baseline = engine.prepare_event(row)["own_events"]
            with patch.object(dev_inputs, "encode_union", compact_union(dev_inputs.encode_union)):
                optimized = engine.prepare_event(row)["own_events"]
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
    receipt_path = OUT / f"EXACT_DEV32_EXECUTION_{os.getpid()}.json"
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
            patch.object(dev_inputs, "encode_union", compact_union(dev_inputs.encode_union)),
            patch.object(dev32_fusion_states, "admission", bounded_admission),
        ):
            dev32_fusion_states.main()
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
