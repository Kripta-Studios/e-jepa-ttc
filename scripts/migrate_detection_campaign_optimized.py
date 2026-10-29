"""Create a CPU-only, immutable checkpoint migration bundle.

This command does not stop a producer, open CUDA, or modify its source run.  It
requires explicit checkpoint hashes and a separately generated CPU
preregistration for the optimized runtime.
"""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.campaign_checkpoint_migration import (
    checkpoint_selection_identity,
    migrate_checkpoint_provenance,
    runtime_provenance_from_file,
    stable_copy_checkpoint,
    validate_checkpoint_envelope,
    verify_evaluation_boundary,
    verify_runtime_transition,
    verify_snapshot,
    verify_source_state_receipt,
)
from e_jepa_ttc.detection.campaign_training import sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    torch_load = cast(Callable[..., object], vars(torch)["load"])
    payload = torch_load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint root must be a mapping")
    return cast(Mapping[str, object], payload)


def _verified_unfinished_parent_chain(
    *,
    old_run: Path,
    source_state: Mapping[str, object],
    old_runtime_prereg: Mapping[str, object],
    old_provenance: Mapping[str, object],
    checkpoint_update: int,
) -> dict[str, object]:
    """Recover and hash-bind ancestry for an intentionally stopped run."""
    path = old_run / "preregistration.json"
    prereg = _load_json(path)
    chain = prereg.get("parent_run_chain")
    start_update = prereg.get("start_update")
    if (
        prereg.get("schema") != "eap_detection_campaign_run_v1"
        or prereg.get("status") != "RUNNING"
        or prereg.get("run_name") != old_run.name
        or prereg.get("config_sha256") != old_runtime_prereg["config_sha256"]
        or prereg.get("evaluation_protocol_sha256")
        != old_runtime_prereg["protocol_sha256"]
        or prereg.get("runtime_provenance") != dict(old_provenance)
        or Path(str(prereg.get("config"))).resolve()
        != Path(str(source_state["config_path"])).resolve()
        or isinstance(start_update, bool)
        or not isinstance(start_update, int)
        or not 0 <= start_update <= checkpoint_update
        or not isinstance(chain, list)
        or any(not isinstance(item, dict) for item in chain)
    ):
        raise ValueError("Unfinished run preregistration identity/chain differs")
    verified_chain: list[dict[str, object]] = []
    for raw_entry in chain:
        entry = cast(dict[str, object], raw_entry)
        summary_path = Path(str(entry.get("summary_path"))).resolve(strict=True)
        if sha256_file(summary_path) != entry.get("summary_sha256"):
            raise ValueError("Inherited parent summary bytes differ")
        summary = _load_json(summary_path)
        if (
            summary.get("status") != entry.get("status")
            or summary.get("final_update") != entry.get("final_update")
            or summary.get("selected_checkpoint") != entry.get("selected_checkpoint")
        ):
            raise ValueError("Inherited parent summary semantics differ")
        selected = entry.get("selected_checkpoint")
        if isinstance(selected, Mapping) and selected.get("status") == "AVAILABLE":
            selected_path = Path(str(selected.get("path"))).resolve(strict=True)
            if sha256_file(selected_path) != selected.get("sha256"):
                raise ValueError("Inherited selected checkpoint bytes differ")
        verified_chain.append(dict(entry))
    return {
        "preregistration_path": str(path.resolve()),
        "preregistration_sha256": sha256_file(path),
        "start_update": start_update,
        "parent_run_chain": verified_chain,
        "status": "PASS_UNFINISHED_INHERITED_CHAIN",
    }


def _migration_parent_summary(
    *,
    old_run: Path,
    output: Path,
    boundary: Mapping[str, object],
    migrated_last: Mapping[str, object],
    migrated_selected: Mapping[str, object] | None,
    old_preregistration_sha256: object,
    old_source_receipt_sha256: object,
    source_state: Mapping[str, object],
    recovery_boundary_sha256: str,
    unfinished_chain: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the exact parent summary consumed by the production resume runner."""
    source_summary_path = old_run / "summary.json"
    source_summary: dict[str, Any] | None = None
    if source_summary_path.is_file():
        source_summary = _load_json(source_summary_path)
        source_chain = source_summary.get("parent_run_chain", [])
        if not isinstance(source_chain, list) or any(
            not isinstance(item, dict) for item in source_chain
        ):
            raise ValueError("Source parent_run_chain is malformed")
        if source_summary.get("final_update") != migrated_last["update"]:
            raise ValueError("Source summary update differs from the pinned last checkpoint")
        source_selected = source_summary.get("selected_checkpoint")
        chain = list(source_chain)
        chain.append(
            {
                "summary_path": str(source_summary_path.resolve()),
                "summary_sha256": sha256_file(source_summary_path),
                "status": source_summary.get("status"),
                "final_update": source_summary.get("final_update"),
                "selected_checkpoint": source_selected,
            }
        )
    elif source_state.get("source_stop_status") == (
        "INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION"
    ):
        if not isinstance(unfinished_chain, Mapping) or not isinstance(
            unfinished_chain.get("parent_run_chain"), list
        ):
            raise ValueError("Intentional stop lacks its verified inherited chain")
        chain = list(cast(list[dict[str, object]], unfinished_chain["parent_run_chain"]))
        chain.append(
            {
                "summary_path": None,
                "summary_sha256": None,
                "status": "RECOVERY_BOUNDARY_FROM_INTENTIONAL_STOP",
                "final_update": migrated_last["update"],
                "selected_checkpoint": None,
            }
        )
    else:
        raise ValueError("Completed source segment lacks its bound summary")
    if migrated_last["best_metric"] is None:
        if migrated_selected is not None:
            raise ValueError("Migrated selected checkpoint exists without a best metric")
    elif (
        migrated_selected is None
        or migrated_selected["best_metric"] != migrated_last["best_metric"]
        or migrated_selected["update"] != boundary.get("selected_update")
    ):
        raise ValueError("Migrated selected sibling differs from the evaluated boundary")
    migrated_selected_path = output / "migrated/selected.pt"
    selected_summary = (
        {
            "status": "AVAILABLE",
            "path": str(migrated_selected_path.resolve()),
            "sha256": migrated_selected["destination_full_sha256"],
            "best_metric": migrated_selected["best_metric"],
        }
        if migrated_selected is not None
        else {
            "status": "NOT_AVAILABLE_NO_SCHEDULED_SELECTION",
            "path": None,
            "sha256": None,
            "best_metric": None,
        }
    )
    return {
        "schema": "eap_detection_campaign_migration_parent_summary_v1",
        "status": "CPU_MIGRATED_GPU_ADMISSION_NOT_RUN",
        "final_update": migrated_last["update"],
        "best_selection_metric": migrated_last["best_metric"],
        "selected_checkpoint": selected_summary,
        "parent_run_chain": chain,
        "migration_boundary": dict(boundary),
        "migration_lineage": {
            "source_summary_path": (
                str(source_summary_path.resolve()) if source_summary is not None else None
            ),
            "source_summary_sha256": (
                sha256_file(source_summary_path) if source_summary is not None else None
            ),
            "source_runtime_preregistration_sha256": old_preregistration_sha256,
            "source_receipt_sha256": old_source_receipt_sha256,
            "source_state_receipt_sha256": source_state["receipt_sha256"],
            "migrated_last_sha256": migrated_last["destination_full_sha256"],
            "migrated_selected_sha256": (
                migrated_selected["destination_full_sha256"]
                if migrated_selected is not None
                else None
            ),
            "recovery_boundary_sha256": recovery_boundary_sha256,
            "unfinished_preregistration_sha256": (
                unfinished_chain.get("preregistration_sha256")
                if unfinished_chain is not None
                else None
            ),
        },
    }


def _preregistration_contract(
    old_run: Path,
    target_preregistration_dir: Path,
    performance_config_relative: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    old_path = old_run / "runtime_preregistration.json"
    target_path = target_preregistration_dir / "runtime_preregistration.json"
    old = _load_json(old_path)
    target = _load_json(target_path)
    transition = verify_runtime_transition(
        old,
        target,
        performance_config_relative=performance_config_relative,
    )
    old_code = cast(dict[str, str], old["code_sha256"])
    target_code = cast(dict[str, str], target["code_sha256"])
    old_snapshot = verify_snapshot(old_run / "code", old_code)
    target_snapshot = verify_snapshot(target_preregistration_dir / "code", target_code)
    old_provenance = runtime_provenance_from_file(old_path, old)
    target_provenance = runtime_provenance_from_file(target_path, target)
    preflight_path = target_preregistration_dir / "preflight_receipt.json"
    preflight = _load_json(preflight_path)
    if (
        preflight.get("schema") != "eap_detection_campaign_cpu_preregistration_v1"
        or preflight.get("status") != "PASS"
        or preflight.get("cuda_or_model_constructed") is not False
        or preflight.get("runtime_provenance") != target_provenance
        or preflight.get("runtime_preregistration_sha256")
        != target_provenance["runtime_preregistration_sha256"]
    ):
        raise ValueError("Target CPU preregistration receipt is inconsistent")
    evidence = {
        "parent_runtime_preregistration_sha256": sha256_file(old_path),
        "parent_source_receipt_sha256": old["source_receipt_sha256"],
        "target_runtime_preregistration_sha256": sha256_file(target_path),
        "target_preflight_receipt_sha256": sha256_file(preflight_path),
        "parent_snapshot": old_snapshot,
        "target_snapshot": target_snapshot,
        "transition": transition,
    }
    return old, target, old_provenance, target_provenance, evidence


def build_migration_bundle(
    *,
    old_run: Path,
    target_preregistration_dir: Path,
    output: Path,
    expected_last_sha256: str,
    expected_selected_sha256: str | None,
    performance_config_relative: str,
    source_state_receipt: Path,
) -> dict[str, object]:
    """Build a write-once bundle while retaining every failure artifact."""
    if output.exists():
        raise FileExistsError(f"Use a new migration output directory: {output}")
    output.mkdir(parents=True)
    try:
        source_state = verify_source_state_receipt(
            source_state_receipt, source_run=old_run
        )
        old, target, old_provenance, target_provenance, contract = (
            _preregistration_contract(
                old_run,
                target_preregistration_dir,
                performance_config_relative,
            )
        )
        pinned_last = output / "pinned/last.pt"
        last_copy = stable_copy_checkpoint(
            old_run / "checkpoints/last.pt",
            pinned_last,
            expected_source_sha256=expected_last_sha256,
        )
        selected_identity: dict[str, object] | None = None
        selected_copy: dict[str, object] | None = None
        pinned_selected = output / "pinned/selected.pt"
        selected_path = old_run / "checkpoints/selected.pt"
        if expected_selected_sha256 is None:
            if selected_path.exists():
                raise ValueError("Selected checkpoint SHA is required at this boundary")
        else:
            selected_copy = stable_copy_checkpoint(
                selected_path,
                pinned_selected,
                expected_source_sha256=expected_selected_sha256,
            )
            selected = _load_checkpoint(pinned_selected)
            if selected["runtime_provenance"] != old_provenance:
                raise ValueError("Parent selected checkpoint provenance differs")
            selected_identity = checkpoint_selection_identity(selected)
            del selected

        last = _load_checkpoint(pinned_last)
        validate_checkpoint_envelope(last)
        if last["runtime_provenance"] != old_provenance:
            raise ValueError("Parent last checkpoint provenance differs from preregistration")
        if (
            last["config_sha256"] != old["config_sha256"]
            or last["protocol_sha256"] != old["protocol_sha256"]
        ):
            raise ValueError("Parent last checkpoint config/protocol identity differs")
        if selected_identity is None and last["best_metric"] is not None:
            raise ValueError("Selected checkpoint SHA is required at this boundary")
        unfinished_chain = (
            _verified_unfinished_parent_chain(
                old_run=old_run,
                source_state=source_state,
                old_runtime_prereg=old,
                old_provenance=old_provenance,
                checkpoint_update=cast(int, last["update"]),
            )
            if source_state.get("source_stop_status")
            == "INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION"
            else None
        )

        evaluation = old["config"]
        if not isinstance(evaluation, Mapping):
            raise ValueError("Parent resolved config is malformed")
        evaluation = evaluation.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("Parent evaluation config is malformed")
        boundary = verify_evaluation_boundary(
            old_run,
            checkpoint=last,
            selected=selected_identity,
            evaluation_updates=tuple(cast(list[int], evaluation["updates"])),
            full_selection_updates=tuple(
                cast(list[int], evaluation["full_selection_updates"])
            ),
            selection_metric=str(evaluation["selection_metric"]),
        )
        del last

        migrated_last = migrate_checkpoint_provenance(
            pinned_last,
            output / "migrated/last.pt",
            old_provenance=old_provenance,
            new_provenance=target_provenance,
        )
        migrated_selected: dict[str, object] | None = None
        if selected_identity is not None:
            migrated_selected = migrate_checkpoint_provenance(
                pinned_selected,
                output / "migrated/selected.pt",
                old_provenance=old_provenance,
                new_provenance=target_provenance,
            )
        recovery_boundary = {
            "schema": "eap_detection_campaign_recovery_boundary_v1",
            "status": "PASS_CPU_EXACT_STATE_GPU_ADMISSION_NOT_RUN",
            "source_state": source_state,
            "source_copies": {"last": last_copy, "selected": selected_copy},
            "runtime_contract": contract,
            "evaluation_boundary": boundary,
            "migrations": {
                "last": migrated_last,
                "selected": migrated_selected,
            },
            "unfinished_inherited_chain": unfinished_chain,
        }
        _write_once(output / "recovery_boundary.json", recovery_boundary)
        parent_summary = _migration_parent_summary(
            old_run=old_run,
            output=output,
            boundary=boundary,
            migrated_last=migrated_last,
            migrated_selected=migrated_selected,
            old_preregistration_sha256=contract[
                "parent_runtime_preregistration_sha256"
            ],
            old_source_receipt_sha256=contract["parent_source_receipt_sha256"],
            source_state=source_state,
            recovery_boundary_sha256=sha256_file(output / "recovery_boundary.json"),
            unfinished_chain=unfinished_chain,
        )
        _write_once(output / "summary.json", parent_summary)
        receipt: dict[str, object] = {
            "schema": "eap_detection_campaign_checkpoint_migration_v1",
            "status": "PASS_CPU_BUNDLE_GPU_ADMISSION_NOT_RUN",
            "source_run": str(old_run.resolve()),
            "target_preregistration_dir": str(target_preregistration_dir.resolve()),
            "performance_config_relative": performance_config_relative,
            "source_copies": {"last": last_copy, "selected": selected_copy},
            "source_state": source_state,
            "contract": contract,
            "evaluation_boundary": boundary,
            "migrations": {
                "last": migrated_last,
                "selected": migrated_selected,
            },
            "target_runtime_provenance": target_provenance,
            "resume_parent_summary_sha256": sha256_file(output / "summary.json"),
            "recovery_boundary_sha256": sha256_file(
                output / "recovery_boundary.json"
            ),
            "producer_was_stopped": False,
            "cuda_or_model_constructed": False,
            "gpu_admission_complete": False,
        }
        _write_once(output / "migration_receipt.json", receipt)
        return receipt
    except BaseException as error:
        failure = {
            "schema": "eap_detection_campaign_checkpoint_migration_failure_v1",
            "status": "FAIL_RETAINED",
            "error_type": type(error).__name__,
            "error": str(error),
            "producer_was_stopped": False,
            "cuda_or_model_constructed": False,
        }
        failure_path = output / "FAILURE.json"
        if not failure_path.exists():
            _write_once(failure_path, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--target-preregistration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-last-sha256", required=True)
    parser.add_argument("--expected-selected-sha256")
    parser.add_argument("--source-state-receipt", type=Path, required=True)
    parser.add_argument(
        "--performance-config-relative",
        default="configs/detection_campaign/performance/indexed_shared_v1.json",
    )
    args = parser.parse_args()
    build_migration_bundle(
        old_run=args.old_run,
        target_preregistration_dir=args.target_preregistration_dir,
        output=args.output,
        expected_last_sha256=args.expected_last_sha256,
        expected_selected_sha256=args.expected_selected_sha256,
        performance_config_relative=args.performance_config_relative,
        source_state_receipt=args.source_state_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
