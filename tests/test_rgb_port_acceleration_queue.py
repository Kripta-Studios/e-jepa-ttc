from __future__ import annotations

from pathlib import Path

import pytest
import torch

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.run import _module_matches
from operational.rgb_port_acceleration.contracts import (
    EXCLUDED_ROUTES,
    FREEZE_SCHEMA,
    SHAPE,
    build_acceleration_freeze,
    canonical_sha256,
)
from operational.rgb_port_acceleration.producer import verify_acceleration_freeze
from operational.rgb_port_acceleration.queue import (
    ACCELERATED_MODULE,
    _augmented_config,
    _route_command,
    _runtime_members,
    _validate_runtime_receipts,
)


def test_only_exact_producer_modules_are_routed(tmp_path: Path) -> None:
    freeze = tmp_path / "ACCELERATION_FREEZE.json"
    freeze.write_text("{}", encoding="utf-8")

    def original(command: list[object], _repository: Path) -> list[str]:
        return list(map(str, command))

    base = ["python", "-m", "operational.rgb_port.train_producers", "--fit-id"]
    routed = _route_command(original, freeze, [*base, "E_A5_MATCHED", "--device", "cuda"], tmp_path)
    assert routed[:3] == ["python", "-m", ACCELERATED_MODULE]
    assert routed[3:6] == ["--acceleration-freeze", str(freeze.resolve()), "--"]
    assert routed[6:] == ["--fit-id", "E_A5_MATCHED", "--device", "cuda"]
    for fit_id in ("E_C2F_MATCHED", "R_A5", "R_C2F"):
        command = [*base, fit_id, "--device", "cuda"]
        routed = _route_command(original, freeze, command, tmp_path)
        assert routed[2] == ACCELERATED_MODULE
        assert routed[-5:] == ["--", "--fit-id", fit_id, "--device", "cuda"]
    unrelated = [*base, "PAIR_E_MATCHED", "--device", "cuda"]
    assert _route_command(original, freeze, unrelated, tmp_path) == unrelated
    assert _route_command(
        original,
        freeze,
        ["python", "-m", "client.operational.rgb_port.train_producers", "--fit-id", "E_A5_MATCHED"],
        tmp_path,
    ) == [
        "python",
        "-m",
        "client.operational.rgb_port.train_producers",
        "--fit-id",
        "E_A5_MATCHED",
    ]


def test_augmented_config_adds_exact_owner_marker_and_bundle_evidence(tmp_path: Path) -> None:
    run = tmp_path / "run"
    receipts = run / "fits/E_A5_MATCHED/acceleration_checkpoints"
    receipts.mkdir(parents=True)
    atomic_write_json(receipts / "checkpoint_006200.json", {"x": 1})
    config = {
        "run_root": str(run),
        "resources": {"heavy_command_markers": [], "project_command_markers": []},
        "package": {"members": []},
    }
    value = _augmented_config(config)
    assert ACCELERATED_MODULE in value["resources"]["heavy_command_markers"]
    assert ACCELERATED_MODULE in value["resources"]["project_command_markers"]
    assert "repo-tree:operational/rgb_port_acceleration" in value["package"]["members"]
    assert (
        "fits/E_A5_MATCHED/acceleration_checkpoints/checkpoint_006200.json"
        in value["package"]["members"]
    )
    assert config["resources"]["heavy_command_markers"] == []
    assert _module_matches(["python", "-m", ACCELERATED_MODULE], [ACCELERATED_MODULE])
    assert not _module_matches(["pytest", ACCELERATED_MODULE], [ACCELERATED_MODULE])


def _frozen_fixture(tmp_path: Path, *, completed: int) -> Path:
    run = tmp_path / "run"
    fit = run / "fits/E_A5_MATCHED"
    versions = fit / "checkpoint_versions"
    versions.mkdir(parents=True)
    source = tmp_path / "sidecar.py"
    backend = tmp_path / "torch_backend.py"
    original = tmp_path / "SOURCE_FREEZE.json"
    admission = tmp_path / "admission.json"
    for path, payload in (
        (source, b"sidecar"),
        (backend, b"backend"),
        (original, b"v2"),
        (admission, b"admission"),
    ):
        path.write_bytes(payload)
    origin = versions / "checkpoint_006189.pt"
    origin.write_bytes(b"origin")
    live = versions / f"checkpoint_{completed:06d}.pt"
    if completed != 6189:
        live.write_bytes(b"accelerated")
    else:
        live = origin
    alias = fit / "checkpoint_last.pt"
    alias.write_bytes(live.read_bytes())
    atomic_write_json(
        fit / "CHECKPOINT_RECEIPT.json",
        {"completed_updates": completed, "checkpoint_sha256": sha256_file(alias)},
    )
    atomic_write_json(
        fit / "CHECKPOINT_POINTER.json",
        {
            "completed_updates": completed,
            "version": live.name,
            "checkpoint_sha256": sha256_file(live),
        },
    )
    freeze = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN",
        "fit_id": "E_A5_MATCHED",
        "graph_fit_id": "E_A5_MATCHED",
        "metrics_fit_ids": ["E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F"],
        "backend": "cudagraphs",
        "admitted_shape": SHAPE,
        "excluded_routes": EXCLUDED_ROUTES,
        "original_source_freeze_path": str(original),
        "original_source_freeze_sha256": sha256_file(original),
        "admission_path": str(admission),
        "admission_sha256": sha256_file(admission),
        "origin": {
            "completed_updates": 6189,
            "checkpoint_path": str(origin),
            "checkpoint_sha256": sha256_file(origin),
            "identity_sha256": "fit-identity",
        },
        "source_sha256": {str(source): sha256_file(source)},
        "orchestration_source_sha256": {str(source): sha256_file(source)},
        "torch_backend_source_sha256": {str(backend): sha256_file(backend)},
    }
    freeze["identity_sha256"] = canonical_sha256(freeze)
    atomic_write_json(run / "ACCELERATION_FREEZE.json", freeze)
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    pointer["identity_sha256"] = "fit-identity"
    atomic_write_json(fit / "CHECKPOINT_POINTER.json", pointer)
    return run


def test_native_origin_is_allowed_but_accelerated_progress_requires_receipts(
    tmp_path: Path,
) -> None:
    origin_run = _frozen_fixture(tmp_path / "origin", completed=6189)
    assert (
        _validate_runtime_receipts(origin_run, require_endpoint=False)["fit_id"] == "E_A5_MATCHED"
    )

    accelerated = _frozen_fixture(tmp_path / "accelerated", completed=6200)
    with pytest.raises(FileNotFoundError, match="runtime or pending"):
        _validate_runtime_receipts(accelerated, require_endpoint=False)
    fit = accelerated / "fits/E_A5_MATCHED"
    freeze = read_json_shared(accelerated / "ACCELERATION_FREEZE.json")
    checkpoint_sha = sha256_file(fit / "checkpoint_last.pt")
    pending_dir = fit / "acceleration_checkpoints"
    pending_dir.mkdir()
    atomic_write_json(
        pending_dir / "PENDING_EXECUTION_RECEIPT.json",
        {
            "schema": "rgb_port_acceleration_pending_v1",
            "status": "PENDING",
            "fit_id": "E_A5_MATCHED",
            "mode": "graph_and_batched_metrics",
            "end_update": 6200,
            "checkpoint_version": "checkpoint_006200.pt",
            "checkpoint_identity_sha256": "fit-identity",
            "acceleration_freeze_sha256": freeze["identity_sha256"],
            "acceleration_freeze_file_sha256": sha256_file(
                accelerated / "ACCELERATION_FREEZE.json"
            ),
            "admission_sha256": freeze["admission_sha256"],
            "source_sha256": freeze["source_sha256"],
            "torch_backend_source_sha256": freeze["torch_backend_source_sha256"],
        },
    )
    assert (
        _validate_runtime_receipts(accelerated, require_endpoint=False)["fit_id"] == "E_A5_MATCHED"
    )
    (pending_dir / "PENDING_EXECUTION_RECEIPT.json").unlink()
    atomic_write_json(
        fit / "ACCELERATION_RUNTIME.json",
        {
            "schema": "rgb_port_acceleration_runtime_v1",
            "status": "ACTIVE",
            "fit_id": "E_A5_MATCHED",
            "mode": "graph_and_batched_metrics",
            "optimizer_updates": 0,
            "completed_updates": 6200,
            "checkpoint_sha256": checkpoint_sha,
            "acceleration_freeze_sha256": freeze["identity_sha256"],
            "acceleration_freeze_file_sha256": sha256_file(
                accelerated / "ACCELERATION_FREEZE.json"
            ),
            "admission_sha256": freeze["admission_sha256"],
            "backend": "cudagraphs",
            "shape": SHAPE,
            "source_sha256": freeze["source_sha256"],
            "torch_backend_source_sha256": freeze["torch_backend_source_sha256"],
        },
    )
    receipt_dir = fit / "acceleration_checkpoints"
    runtime_archive = receipt_dir / "runtime_006200.json"
    atomic_write_json(runtime_archive, read_json_shared(fit / "ACCELERATION_RUNTIME.json"))
    runtime_sha = sha256_file(runtime_archive)
    atomic_write_json(
        receipt_dir / "checkpoint_006200.json",
        {
            "schema": "rgb_port_acceleration_checkpoint_receipt_v1",
            "status": "COMPLETE",
            "fit_id": "E_A5_MATCHED",
            "mode": "graph_and_batched_metrics",
            "completed_updates": 6200,
            "checkpoint_version": "checkpoint_006200.pt",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_identity_sha256": "fit-identity",
            "acceleration_freeze_sha256": freeze["identity_sha256"],
            "acceleration_freeze_file_sha256": sha256_file(
                accelerated / "ACCELERATION_FREEZE.json"
            ),
            "runtime_receipt_sha256": runtime_sha,
            "runtime_snapshot": runtime_archive.name,
            "admission_sha256": freeze["admission_sha256"],
            "source_sha256": freeze["source_sha256"],
            "torch_backend_source_sha256": freeze["torch_backend_source_sha256"],
            "backend": "cudagraphs",
            "shape": SHAPE,
        },
    )
    with pytest.raises(RuntimeError, match="prewarm"):
        _validate_runtime_receipts(accelerated, require_endpoint=False)
    atomic_write_json(
        fit / "PREWARM_QA.json",
        {
            "schema": "rgb_port_acceleration_prewarm_qa_v1",
            "status": "PASSED",
            "fit_id": "E_A5_MATCHED",
            "completed_updates": 6189,
            "optimizer_updates": 0,
            "checkpoint_sha256": freeze["origin"]["checkpoint_sha256"],
            "checkpoint_identity_sha256": "fit-identity",
            "acceleration_freeze_sha256": freeze["identity_sha256"],
            "acceleration_freeze_file_sha256": sha256_file(
                accelerated / "ACCELERATION_FREEZE.json"
            ),
            "runtime_receipt_sha256": runtime_sha,
            "post_warm_exact": {"full_state": True},
            "counters": {},
        },
    )
    assert (
        _validate_runtime_receipts(accelerated, require_endpoint=True)["fit_id"] == "E_A5_MATCHED"
    )
    runtime = read_json_shared(fit / "ACCELERATION_RUNTIME.json")
    runtime["status"] = "READY"
    runtime["post_warm_exact"] = {"restored_again": True}
    atomic_write_json(fit / "ACCELERATION_RUNTIME.json", runtime)
    assert (
        _validate_runtime_receipts(accelerated, require_endpoint=True)["fit_id"] == "E_A5_MATCHED"
    )


def test_runtime_members_are_stable_for_orphan_adoption(tmp_path: Path) -> None:
    run = tmp_path / "run"
    first = _runtime_members(run)
    second = _runtime_members(run)
    assert first == second


def test_builder_contract_is_accepted_by_runtime_and_copies_origin(tmp_path: Path) -> None:
    run = tmp_path / "run"
    fit = run / "fits/E_A5_MATCHED"
    versions = fit / "checkpoint_versions"
    versions.mkdir(parents=True)
    (run / "SOURCE_FREEZE.json").write_text('{"version": 2}\n', encoding="utf-8")
    checkpoint = versions / "checkpoint_006189.pt"
    checkpoint.write_bytes(b"sealed full state")
    digest = sha256_file(checkpoint)
    identity = "fit-identity"
    atomic_write_json(
        fit / "CHECKPOINT_POINTER.json",
        {
            "completed_updates": 6189,
            "version": checkpoint.name,
            "checkpoint_sha256": digest,
            "identity_sha256": identity,
        },
    )
    atomic_write_json(
        fit / "CHECKPOINT_RECEIPT.json",
        {
            "completed_updates": 6189,
            "checkpoint_sha256": digest,
            "identity_sha256": identity,
        },
    )
    admission = tmp_path / "admission.json"
    atomic_write_json(
        admission,
        {
            "schema": "rgb_port_e_a5_cudagraph_admission_v1",
            "status": "PASSED",
            "fit_id": "E_A5_MATCHED",
            "backend": "cudagraphs",
            "completed_updates": 6189,
            "next_cursor": {"shape": [32, 3]},
            "compatible": True,
            "checkpoint_unchanged": True,
            "optimizer_updates": 0,
            "bindings": {
                "source_freeze_sha256": sha256_file(run / "SOURCE_FREEZE.json"),
                "checkpoint_sha256": digest,
                "checkpoint_identity_sha256": identity,
                "train_producers_sha256": sha256_file(
                    Path("operational/rgb_port/train_producers.py")
                ),
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
            },
        },
    )
    value = build_acceleration_freeze(run, admission)
    assert verify_acceleration_freeze(run / "ACCELERATION_FREEZE.json") == value
    origin_copy = run / "acceleration/E_A5_ORIGIN_006189.pt"
    assert sha256_file(origin_copy) == digest
    assert value["origin"]["checkpoint_path"] == str(origin_copy.resolve())
    assert set(value["source_sha256"]) == {
        str(Path("operational/rgb_port_acceleration/producer.py").resolve()),
        str(Path("operational/rgb_port/train_producers.py").resolve()),
    }
    assert len(value["torch_backend_source_sha256"]) == 2
