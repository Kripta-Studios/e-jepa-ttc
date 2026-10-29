"""Archive superseded same-update receipts; retain native pending recovery gates.

Specific to the interrupted 26338 pause transaction. Never edits checkpoint bytes,
the native pointer/journal, source code, freeze manifests, or pending proofs.
The original checkpoint bytes bound by the superseded receipts are unavailable;
this procedure makes no claim of tensor parity with that old serialization.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import psutil
import torch

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port_acceleration.contracts import validate_acceleration_freeze
from operational.rgb_port_acceleration.queue import _validate_runtime_receipts
from operational.rgb_port_c2f_graph.contracts import (
    validate_c2f_graph_freeze,
    validate_c2f_graph_lineage,
)
from operational.rgb_port_concurrent.contracts import (
    validate_concurrent_freeze,
    validate_concurrent_lineage,
)
from operational.rgb_port_io_recovery.common import validate_freeze
from operational.rgb_port_pipeline_v2.receipts import validate_fit_lineage

CURRENT_SHA = "36485f9881182bbf0e57da29563a396585c9d6e550fd29e5cd8bba88b35cdcaa"
OLD_SHA = "ffddb3a1a7b9f08f5c9bc5723baa79f2bf47250a1929b41c4b285436c6211f41"
FIT = "E_A5_MATCHED"
UPDATE = 26338


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    run = output.parent
    fit = run / "fits" / FIT
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if process.pid == os.getpid():
            continue
        require(
            not any(
                token.startswith("operational.rgb_port")
                and token.endswith((".queue", ".producer", ".train_producers", ".run"))
                for token in command
            ),
            f"Training/supervisor still live: {process.pid}",
        )
    validate_freeze(run)
    validate_acceleration_freeze(run / "ACCELERATION_FREEZE.json")
    validate_concurrent_freeze(run / "CONCURRENT_FREEZE.json")
    validate_c2f_graph_freeze(run / "C2F_GRAPH_FREEZE.json")
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    native = read_json_shared(fit / "CHECKPOINT_RECEIPT.json")
    journal = read_json_shared(fit / "UPDATE_JOURNAL.json")
    version = fit / "checkpoint_versions" / "checkpoint_026338.pt"
    for member in (version, fit / "checkpoint_last.pt"):
        require(sha256_file(member) == CURRENT_SHA, "Native checkpoint bytes changed")
    require(
        pointer["checkpoint_sha256"] == native["checkpoint_sha256"] == CURRENT_SHA,
        "Native hash receipts disagree",
    )
    require(
        pointer["completed_updates"]
        == native["completed_updates"]
        == journal["completed_updates"]
        == journal["durable_updates"]
        == UPDATE,
        "Native update receipts disagree",
    )
    require(
        journal["pending_update_upper"] == 0
        and native["accumulation_index"] == 0
        and native["complete_state"]
        and native["status"] == "PAUSED_RESOURCE",
        "Native checkpoint is not a complete safe pause",
    )
    payload = torch.load(version, map_location="cpu", weights_only=False)
    require(
        payload["completed_updates"] == UPDATE and payload["accumulation_index"] == 0,
        "Payload is not at the durable optimizer boundary",
    )
    require(
        payload["identity_sha256"]
        == pointer["identity_sha256"]
        == native["identity_sha256"]
        == journal["identity_sha256"],
        "Identity differs",
    )
    require(
        {int(v["step"]) for v in payload["optimizer_state_dict"]["state"].values()} == {UPDATE},
        "Adam steps differ",
    )
    require(
        payload["code_migration_sha256"] == sha256_file(run / "AUDIT_CODE_MIGRATION.json"),
        "Audited code migration differs",
    )
    require(
        all(
            key in payload
            for key in (
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "cursor",
                "sampler_generator_state",
                "torch_rng_state",
                "cuda_rng_state_all",
                "numpy_random_state",
                "python_random_state",
            )
        ),
        "Incomplete full-state payload",
    )
    preserved = [
        version,
        fit / "checkpoint_last.pt",
        fit / "CHECKPOINT_POINTER.json",
        fit / "CHECKPOINT_RECEIPT.json",
        fit / "UPDATE_JOURNAL.json",
    ]
    stale = []
    for layer in ("acceleration", "concurrent", "pipeline"):
        root = fit / f"{layer}_checkpoints"
        pending_path = root / "PENDING_EXECUTION_RECEIPT.json"
        pending = read_json_shared(pending_path)
        receipt_path = root / "checkpoint_026338.json"
        receipt = read_json_shared(receipt_path)
        require(
            pending["status"] == "PENDING"
            and pending["start_update"] == pending["end_update"] == UPDATE,
            "Not a same-update pending transaction",
        )
        require(
            pending["checkpoint_identity_sha256"] == pointer["identity_sha256"]
            and pending["checkpoint_version"] == pointer["version"]
            and pending["fit_id"] == FIT,
            "Pending proof identity differs",
        )
        require(
            receipt["checkpoint_sha256"] == OLD_SHA and receipt["completed_updates"] == UPDATE,
            "Not the known stale receipt",
        )
        for key, value in pending.items():
            if key not in {"schema", "status", "start_update", "end_update"}:
                require(receipt.get(key) == value, f"Pending/old receipt binding differs: {key}")
        if layer != "concurrent":
            snapshot = root / ("runtime_026338.json" if layer == "acceleration" else "timings_026338.json")
            sha_key = (
                "runtime_receipt_sha256" if layer == "acceleration" else "timings_sha256"
            )
            require(sha256_file(snapshot) == receipt[sha_key], "Stale snapshot hash differs")
            require(
                read_json_shared(snapshot)["checkpoint_sha256"] == OLD_SHA,
                "Stale snapshot no longer binds old serialization",
            )
            stale.append(snapshot)
        stale.append(receipt_path)
        preserved.append(pending_path)
    pipeline_runtime = fit / "PIPELINE_RUNTIME.json"
    runtime = read_json_shared(pipeline_runtime)
    require(runtime.get("last_checkpoint_sha256") == OLD_SHA
            and runtime.get("last_saved_update") == UPDATE
            and runtime.get("fit_id") == FIT,
            "Pipeline runtime is not the known superseded same-update state")
    stale.append(pipeline_runtime)
    protected = {str(p.relative_to(run)): sha256_file(p) for p in preserved}
    inventory = {str(p.relative_to(fit)): sha256_file(p) for p in stale}
    evidence = {
        "status": "VALIDATED_ONLY",
        "utc": datetime.now(UTC).isoformat(),
        "current_checkpoint_sha256": CURRENT_SHA,
        "superseded_checkpoint_sha256": OLD_SHA,
        "completed_updates": UPDATE,
        "optimizer_updates_consumed": 0,
        "gpu_seconds": 0,
        "source_sha256": sha256_file(Path(__file__)),
        "protected": protected,
        "superseded_auxiliary_files": inventory,
        "old_checkpoint_tensor_parity_available": False,
        "next_step": "Native recovery from existing pending proofs and exact GPU restore QA",
    }
    atomic_write_json(output / "PRECHECK.json", evidence)
    if args.apply:
        require((run / "PAUSE").is_file(), "Campaign pause must remain owned")
        moved = []
        try:
            for source in stale:
                destination = output / "superseded" / source.relative_to(fit)
                if destination.exists():
                    require(sha256_file(destination) == sha256_file(source), "Existing recovery archive differs")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                require(sha256_file(destination) == sha256_file(source), "Archive copy differs")
                source.unlink()
                moved.append((source, destination))
            _validate_runtime_receipts(run, require_endpoint=False, fit_id=FIT)
            validate_concurrent_lineage(run, run / "CONCURRENT_FREEZE.json", FIT)
            validate_c2f_graph_lineage(run, run / "C2F_GRAPH_FREEZE.json")
            validate_fit_lineage(run, read_json_shared(run / "PIPELINE_FREEZE.json"), run / "PIPELINE_FREEZE.json", FIT)
            require(
                protected == {str(p.relative_to(run)): sha256_file(p) for p in preserved},
                "Protected checkpoint/pending proof changed",
            )
        except Exception:
            for source, destination in reversed(moved):
                shutil.copy2(destination, source)
            raise
        evidence["status"] = "NATIVE_PENDING_RECOVERY_ADMITTED_GPU_RESTORE_PENDING"
        evidence["upstream_validators_passed"] = True
        atomic_write_json(output / "RESULT.json", evidence)
    print(
        json.dumps(
            {k: evidence[k] for k in ("status", "completed_updates", "optimizer_updates_consumed")}
        )
    )


if __name__ == "__main__":
    main()
