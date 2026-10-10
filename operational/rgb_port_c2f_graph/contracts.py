"""Immutable authorization for the RGB-PORT C2F CUDA-graph overlay."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256
from operational.rgb_port_concurrent.contracts import validate_concurrent_freeze
from operational.rgb_port_revision.migration import inventory_matches

SCHEMA = "rgb_port_c2f_graph_freeze_v2"
FIT_ID = "E_C2F_MATCHED"
SOURCE_FILES = ("contracts.py", "producer.py", "queue.py")
ADMISSION_GATES = {
    *{
        f"{case}_{gate}"
        for case in ("full", "foreground")
        for gate in (
            "gradient_cross_max",
            "compiled_intra_max",
            "compiled_vs_eager_jitter",
            "loss_repeat_max",
            "component_repeat_max",
            "gradients_finite",
        )
    },
    "graph_dispatch_observed",
    "actual_cuda_graph_launch_observed",
    "output_parity",
    "finite",
    "full_state_restore_exact",
    "paused_stop_marker_present",
    "no_live_c2f_trainer",
    "resource_preflight_passed",
    "paused_full_checkpoint_exact",
}


def _repeat_policy_is_exact(policy: object) -> bool:
    requirements = policy.get("requirements", {}) if isinstance(policy, dict) else {}
    failed = policy.get("preserved_exact_repeat_failure", {}) if isinstance(policy, dict) else {}
    return bool(
        isinstance(policy, dict)
        and policy.get("schema") == "rgb_port_c2f_graph_numerical_policy_v2"
        and policy.get("status") == "REGISTERED_BEFORE_MEASUREMENT"
        and policy.get("modes") == ["eager", "compiled"]
        and policy.get("loss_cases") == ["full", "foreground"]
        and policy.get("resets_per_mode_case") == 5
        and policy.get("gradient_relative_l2")
        == {
            "all_cross_pairwise_max_lte": 0.002,
            "compiled_intra_pairwise_max_lte": 0.002,
            "compiled_intra_lte_eager_intra_plus": 0.002,
        }
        and policy.get("gradient_relative_l2_definition")
        == "l2(left-right)/max(l2(left),l2(right),1e-30)"
        and policy.get("loss_component_pairwise_scope")
        == "all_eager_intra_compiled_intra_and_eager_compiled_cross_pairs"
        and policy.get("loss_abs_pairwise_max_lte") == 0.0005
        and policy.get("component_abs_pairwise_max_lte") == 0.0005
        and policy.get("output_abs_lte") == 0.0005
        and requirements.get("actual_cuda_graph_launch") is True
        and requirements.get("every_repeat_gradient_vector_finite") is True
        and requirements.get("full_state_rng_restore_exact") is True
        and requirements.get("optimizer_updates") == 0
        and requirements.get("record_every_repeat") is True
        and requirements.get("localize_discrepant_parameters") is True
        and failed.get("status") == "FAILED"
        and sha256_file(Path(failed.get("path", ""))) == failed.get("sha256")
        and sha256_file(Path(failed.get("executed_source_path", "")))
        == failed.get("executed_source_sha256")
    )


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        str((root / name).resolve(strict=True)): sha256_file(root / name)
        for name in SOURCE_FILES
    }


def build_c2f_graph_freeze(
    run: Path, admission_path: Path, *, output: Path | None = None
) -> dict[str, Any]:
    """Freeze a passed zero-update admission and the current paused C2F checkpoint."""
    run = run.resolve(strict=True)
    concurrent_path = run / "CONCURRENT_FREEZE.json"
    concurrent = validate_concurrent_freeze(concurrent_path)
    admission_path = admission_path.resolve(strict=True)
    admission = read_json_shared(admission_path)
    fit = run / "fits" / FIT_ID
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    version = fit / "checkpoint_versions" / str(pointer.get("version", ""))
    receipt = read_json_shared(fit / "CHECKPOINT_RECEIPT.json")
    journal = read_json_shared(fit / "UPDATE_JOURNAL.json")
    if (
        admission.get("schema") != "rgb_port_c2f_graph_admission_numerical_v2"
        or admission.get("status") != "PASSED"
        or admission.get("fit_id") != FIT_ID
        or admission.get("optimizer_updates") != 0
        or admission.get("shape") != {"batch_size": 32, "frames": 3}
        or admission.get("backend") != "cudagraphs"
        or not _repeat_policy_is_exact(admission.get("repeat_policy"))
        or admission.get("repeat_policy")
        != read_json_shared(Path(admission["repeat_policy_path"]))
        or admission.get("repeat_policy_sha256")
        != sha256_file(Path(admission["repeat_policy_path"]))
        or admission.get("source_freeze_sha256")
        != sha256_file(Path(admission["source_freeze_path"]))
        or set(admission.get("gates", {})) != ADMISSION_GATES
        or not all(admission["gates"].values())
        or sha256_file(Path(admission["admission_source_path"]))
        != admission.get("admission_source_sha256")
        or receipt.get("status") not in {"PAUSED_REQUESTED", "PAUSED_RESOURCE"}
        or journal.get("completed_updates") != pointer.get("completed_updates")
        or journal.get("durable_updates") != pointer.get("completed_updates")
        or journal.get("pending_update_upper") != 0
        or journal.get("recovery_upper") != 0
        or journal.get("identity_sha256") != pointer.get("identity_sha256")
        or pointer.get("checkpoint_sha256") != sha256_file(version)
        or admission.get("checkpoint_sha256") != pointer.get("checkpoint_sha256")
        or admission.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
        or admission.get("completed_updates") != pointer.get("completed_updates")
    ):
        raise ValueError("C2F graph admission does not bind the paused full checkpoint")
    origin_dir = run / "c2f_graph"
    origin_dir.mkdir(parents=True, exist_ok=True)
    origin_copy = origin_dir / f"origin_{int(pointer['completed_updates']):06d}.pt"
    if not origin_copy.exists():
        try:
            os.link(version, origin_copy)
        except OSError:
            shutil.copyfile(version, origin_copy)
    if sha256_file(origin_copy) != pointer["checkpoint_sha256"]:
        raise RuntimeError("Immutable C2F graph origin copy differs")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "fit_id": FIT_ID,
        "baseline_mode": "DISABLED_BASELINE",
        "effective_mode": "CUDAGRAPH_ON",
        "backend": "cudagraphs",
        "shape": {"batch_size": 32, "frames": 3},
        "scientific_arms_added": 0,
        "concurrent_freeze_path": str(concurrent_path.resolve()),
        "concurrent_freeze_sha256": sha256_file(concurrent_path),
        "concurrent_freeze_identity_sha256": concurrent["identity_sha256"],
        "admission_path": str(admission_path),
        "admission_sha256": sha256_file(admission_path),
        "admission_source_path": admission["admission_source_path"],
        "admission_source_sha256": admission["admission_source_sha256"],
        "repeat_policy_path": admission["repeat_policy_path"],
        "repeat_policy_sha256": admission["repeat_policy_sha256"],
        "source_freeze_path": admission["source_freeze_path"],
        "source_freeze_sha256": admission["source_freeze_sha256"],
        "preserved_exact_repeat_failure": admission["preserved_exact_repeat_failure"],
        "source_sha256": _source_hashes(),
        "torch_backend_source_sha256": admission["torch_backend_source_sha256"],
        "origin": {
            "completed_updates": pointer["completed_updates"],
            "checkpoint_sha256": pointer["checkpoint_sha256"],
            "identity_sha256": pointer["identity_sha256"],
            "immutable_checkpoint_path": str(origin_copy.resolve()),
            "immutable_checkpoint_sha256": sha256_file(origin_copy),
        },
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    target = output.resolve() if output else run / "C2F_GRAPH_FREEZE.json"
    if target.exists() and read_json_shared(target) != payload:
        raise RuntimeError("C2F graph freeze changed")
    if not target.exists():
        atomic_write_json(target, payload)
    return payload


def validate_c2f_graph_freeze(path: Path) -> dict[str, Any]:
    """Validate graph admission, parent freeze, sources, backend and origin copy."""
    value = read_json_shared(path)
    if (
        value.get("schema") != SCHEMA
        or value.get("status") != "FROZEN"
        or value.get("fit_id") != FIT_ID
        or value.get("baseline_mode") != "DISABLED_BASELINE"
        or value.get("effective_mode") != "CUDAGRAPH_ON"
        or value.get("backend") != "cudagraphs"
        or value.get("shape") != {"batch_size": 32, "frames": 3}
        or value.get("scientific_arms_added") != 0
        or value.get("identity_sha256")
        != canonical_sha256({k: v for k, v in value.items() if k != "identity_sha256"})
    ):
        raise ValueError("C2F graph freeze scope differs")
    concurrent_path = Path(value["concurrent_freeze_path"])
    concurrent = validate_concurrent_freeze(concurrent_path)
    admission_path = Path(value["admission_path"])
    admission = read_json_shared(admission_path)
    origin = value["origin"]
    if (
        sha256_file(concurrent_path) != value["concurrent_freeze_sha256"]
        or concurrent["identity_sha256"] != value["concurrent_freeze_identity_sha256"]
        or sha256_file(admission_path) != value["admission_sha256"]
        or admission.get("status") != "PASSED"
        or admission.get("optimizer_updates") != 0
        or admission.get("schema") != "rgb_port_c2f_graph_admission_numerical_v2"
        or not _repeat_policy_is_exact(admission.get("repeat_policy"))
        or admission.get("repeat_policy")
        != read_json_shared(Path(admission["repeat_policy_path"]))
        or admission.get("repeat_policy_path") != value["repeat_policy_path"]
        or admission.get("repeat_policy_sha256") != value["repeat_policy_sha256"]
        or sha256_file(Path(value["repeat_policy_path"])) != value["repeat_policy_sha256"]
        or admission.get("source_freeze_path") != value["source_freeze_path"]
        or admission.get("source_freeze_sha256") != value["source_freeze_sha256"]
        or sha256_file(Path(value["source_freeze_path"])) != value["source_freeze_sha256"]
        or admission.get("preserved_exact_repeat_failure")
        != value["preserved_exact_repeat_failure"]
        or sha256_file(Path(value["preserved_exact_repeat_failure"]["path"]))
        != value["preserved_exact_repeat_failure"]["sha256"]
        or set(admission.get("gates", {})) != ADMISSION_GATES
        or not all(admission["gates"].values())
        or sha256_file(Path(value["admission_source_path"]))
        != value["admission_source_sha256"]
        or admission.get("admission_source_path") != value["admission_source_path"]
        or admission.get("admission_source_sha256") != value["admission_source_sha256"]
        or admission.get("checkpoint_sha256") != origin["checkpoint_sha256"]
        or admission.get("completed_updates") != origin["completed_updates"]
        or admission.get("torch_backend_source_sha256")
        != value["torch_backend_source_sha256"]
        or sha256_file(Path(origin["immutable_checkpoint_path"]))
        != origin["immutable_checkpoint_sha256"]
        or origin["immutable_checkpoint_sha256"] != origin["checkpoint_sha256"]
        or not inventory_matches(value.get("source_sha256", {}), _source_hashes(), path.parent)
    ):
        raise RuntimeError("C2F graph freeze bindings changed")
    return value


def validate_c2f_graph_lineage(
    run: Path, freeze_path: Path, *, require_endpoint: bool = False
) -> None:
    """Validate every retained post-origin checkpoint and its graph runtime proof."""
    freeze = validate_c2f_graph_freeze(freeze_path)
    freeze_file_sha = sha256_file(freeze_path)
    fit = run / "fits" / FIT_ID
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    completed = int(pointer.get("completed_updates", -1))
    origin = freeze["origin"]
    version = fit / "checkpoint_versions" / str(pointer.get("version", ""))
    if (
        completed < int(origin["completed_updates"])
        or not version.is_file()
        or pointer.get("checkpoint_sha256") != sha256_file(version)
        or pointer.get("identity_sha256") != origin["identity_sha256"]
    ):
        raise RuntimeError("C2F graph pointer moved outside its frozen lineage")
    if completed == int(origin["completed_updates"]):
        if pointer.get("checkpoint_sha256") != origin["checkpoint_sha256"]:
            raise RuntimeError("C2F graph origin checkpoint changed")
        if require_endpoint:
            raise RuntimeError("C2F graph endpoint has not advanced through the overlay")
        return
    root = fit / "c2f_graph_checkpoints"
    receipt_path = root / f"checkpoint_{completed:06d}.json"
    common = {
        "fit_id": FIT_ID,
        "c2f_graph_freeze_sha256": freeze["identity_sha256"],
        "c2f_graph_freeze_file_sha256": freeze_file_sha,
        "source_sha256": freeze["source_sha256"],
        "torch_backend_source_sha256": freeze["torch_backend_source_sha256"],
        "backend": "cudagraphs",
        "shape": {"batch_size": 32, "frames": 3},
        "optimizer_updates": 0,
    }
    if not receipt_path.is_file():
        pending_path = root / "PENDING_EXECUTION_RECEIPT.json"
        if not pending_path.is_file():
            raise FileNotFoundError("C2F graph progress lacks receipt or pending proof")
        pending = read_json_shared(pending_path)
        if (
            pending.get("schema") != "rgb_port_c2f_graph_pending_v1"
            or pending.get("status") != "PENDING"
            or any(pending.get(name) != item for name, item in common.items())
            or pending.get("end_update") != completed
            or pending.get("checkpoint_version") != pointer.get("version")
            or pending.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
        ):
            raise RuntimeError("C2F graph pending proof differs")
        return
    for checkpoint in (fit / "checkpoint_versions").glob("checkpoint_*.pt"):
        update = int(checkpoint.stem.removeprefix("checkpoint_"))
        if update <= int(origin["completed_updates"]):
            continue
        item_path = root / f"checkpoint_{update:06d}.json"
        if not item_path.is_file():
            raise FileNotFoundError(f"C2F graph receipt absent for {checkpoint.name}")
        item = read_json_shared(item_path)
        snapshot = root / str(item.get("runtime_snapshot", ""))
        if (
            item.get("schema") != "rgb_port_c2f_graph_checkpoint_receipt_v1"
            or item.get("status") not in {"COMPLETE", "RECOVERED_FROM_PENDING"}
            or any(item.get(name) != value for name, value in common.items())
            or item.get("completed_updates") != update
            or item.get("checkpoint_version") != checkpoint.name
            or item.get("checkpoint_sha256") != sha256_file(checkpoint)
            or item.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
            or item.get("runtime_snapshot") != f"runtime_{update:06d}.json"
            or not snapshot.is_file()
            or item.get("runtime_snapshot_sha256") != sha256_file(snapshot)
        ):
            raise RuntimeError(f"C2F graph receipt changed: {update}")
    if require_endpoint:
        native = read_json_shared(fit / "CHECKPOINT_RECEIPT.json")
        if native.get("status") != "COMPLETE" or native.get("completed_updates") != completed:
            raise RuntimeError("C2F graph endpoint is incomplete")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--admission", type=Path, required=True)
    build.add_argument("--output", type=Path)
    check = sub.add_parser("validate")
    check.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    value = (
        build_c2f_graph_freeze(args.run, args.admission, output=args.output)
        if args.action == "build"
        else validate_c2f_graph_freeze(args.freeze)
    )
    print(json.dumps({"identity_sha256": value["identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

