from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from operational.final_refit.eager_probe import (
    CASE_KINDS,
    CONTROL_POLICY,
    CONTROL_POLICY_SHA256,
    FIT_IDS,
    REGIMES,
)
from operational.final_refit.eager_probe import (
    SCHEMA as EAGER_SCHEMA,
)
from operational.final_refit.recipe import canonical_sha256, sha256
from operational.final_refit.runtime_migration import (
    ADMISSION_SCHEMA,
    OPTIMIZATION_ADMISSION_SCHEMA,
    _validate_prior_execution_lineage,
    bind_runtime_resume,
    migrate_runtime_state,
    migrate_runtime_states_v3,
    resolve_runtime_checkpoint_identity,
    resolve_runtime_checkpoint_identity_v3,
    runtime_checkpoint_binding_kwargs,
    runtime_checkpoint_binding_kwargs_v3,
    validate_optimization_admission,
    validate_runtime_batch_migration,
    validate_runtime_binding,
    validate_runtime_migration,
)
from operational.rgb_port.train_producers import ProducerCheckpoint

SOURCE_NAME = "operational/final_refit/runtime_migration.py"
CONFIG_NAME = "configs/rgb_port/producers.json"
STAGING_ROOT = Path(__file__).resolve().parents[2]


def _runtime_source_sha() -> str:
    return sha256(STAGING_ROOT / SOURCE_NAME)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _plan(path: Path, runtime_migration_admission: str | None, source: str) -> dict[str, Any]:
    inputs = {
        "data": "d" * 64,
        "producer_config": sha256(Path(CONFIG_NAME)),
        "runtime_admission": "r" * 64,
    }
    if runtime_migration_admission is not None:
        inputs["runtime_migration_admission"] = runtime_migration_admission
    value: dict[str, Any] = {
        "schema": "final_refit_rgb_producer_plan_v1",
        "status": "FROZEN_AUTHORIZED_PRODUCER_SCOPE",
        "launch_authorized": True,
        "input_sha256": inputs,
        "executed_source_sha256": {SOURCE_NAME: source},
        "stages": [{"fit_id": "R_A5", "updates": 49932, "seed": 7}],
    }
    value["plan_sha256"] = canonical_sha256(value)
    _write_json(path, value)
    return value


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "old" / "fits" / "R_A5"
    source.mkdir(parents=True)
    old_plan_path = tmp_path / "old_plan.json"
    old_plan = _plan(old_plan_path, None, "1" * 64)
    identity = {
        "schema": "final_refit_rgb_producer_identity_v1",
        "fit_id": "R_A5",
        "seed": 7,
        "runtime": {"precision": "bf16", "effective_batch_size": 32},
        "producer_contract_path": str(old_plan_path.resolve()),
        "producer_contract_sha256": sha256(old_plan_path),
        "executed_source_sha256": {
            SOURCE_NAME: "1" * 64,
            CONFIG_NAME: sha256(Path(CONFIG_NAME)),
        },
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    payload = {
        "schema": "rgb_port_producer_checkpoint_v1",
        "code_migration_sha256": None,
        "identity": identity,
        "identity_sha256": identity["identity_sha256"],
        "model_state_dict": {"weight": torch.tensor([[1.0, 2.0]])},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(169)}}},
        "scheduler_state_dict": {"last_epoch": 0},
        "sampler_generator_state": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "torch_rng_state": torch.tensor([4, 5], dtype=torch.uint8),
        "cuda_rng_state_all": None,
        "numpy_random_state": ("MT19937", np.array([7, 8], dtype=np.uint32), 2, 0, 0.0),
        "python_random_state": (3, (1, 2, 3), None),
        "cursor": {
            "epoch": 1,
            "position": 5408,
            "order": torch.tensor([2, 0, 1]),
            "loss_sums": {"total": 1.5},
            "examples": 5408,
            "pending_curve": [{"update": index} for index in range(101, 170)],
        },
        "completed_updates": 169,
        "recovery_upper": 0,
        "accumulation_index": 0,
        "status": "PAUSED_RESOURCE",
    }
    checkpoint = source / "checkpoint_migration_origin_000169.pt"
    torch.save(payload, checkpoint)
    checkpoint_hash = sha256(checkpoint)
    _write_json(
        source / "CHECKPOINT_POINTER.json",
        {
            "schema": "rgb_port_checkpoint_pointer_v1",
            "identity_sha256": identity["identity_sha256"],
            "completed_updates": 169,
            "version": checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    _write_json(
        source / "CHECKPOINT_RECEIPT.json",
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "fit_id": "R_A5",
            "status": "PAUSED_RESOURCE",
            "completed_updates": 169,
            "scientific_endpoint": False,
            "identity_sha256": identity["identity_sha256"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "complete_state": True,
            "accumulation_index": 0,
            "recovery_upper": 0,
        },
    )
    _write_json(
        source / "UPDATE_JOURNAL.json",
        {
            "schema": "rgb_port_update_journal_v1",
            "identity_sha256": identity["identity_sha256"],
            "completed_updates": 169,
            "durable_updates": 169,
            "recovery_upper": 0,
            "pending_update_upper": 0,
        },
    )
    _write_json(
        source / "curve_000100.json",
        {"updates": [{"update": index} for index in range(1, 101)]},
    )
    after_source = _runtime_source_sha()
    admission_path = tmp_path / "admission.json"
    comparison_value = {
        "names_equal": True,
        "shapes_equal": True,
        "finite": True,
        "max_abs": 0.0,
        "relative_l2": 0.0,
    }
    numerical_field = {
        "policy": CONTROL_POLICY["numerical_rule"],
        "strict_original_threshold_passed": True,
        "strict_control_threshold_passed": True,
        "control_dominance_passed": True,
        "passed": True,
        "candidate_reference": comparison_value,
        "reference_reference_control": comparison_value,
    }
    eager_cases = []
    for fit_id in FIT_IDS:
        for regime in REGIMES:
            for kind in CASE_KINDS:
                eager_cases.append(
                    {
                        "fit_id": fit_id,
                        "regime": regime,
                        "case_kind": kind,
                        "batch_parity": True,
                        "initial_state_equal": True,
                        "resume_exact": True,
                        "scientific_updates": 0,
                        "candidate_technical_optimizer_steps": 2,
                        "reference_technical_optimizer_steps": 2,
                        "control_technical_optimizer_steps": 4,
                        "control_policy_sha256": CONTROL_POLICY_SHA256,
                        "passed": True,
                        "comparisons": [
                            {
                                "update": update,
                                "rng_equal": True,
                                "sampler_generator_equal": True,
                                **{
                                    field: numerical_field
                                    for field in (
                                        "loss",
                                        "components",
                                        "raw_gradients",
                                        "clip_norm",
                                        "clipped_gradients",
                                        "model_after_step",
                                        "optimizer_after_step",
                                        "scheduler_after_step",
                                    )
                                },
                            }
                            for update in (1, 2)
                        ],
                    }
                )
    evidence_payloads = {
        "state_parity": {
            "schema": "final_refit_checkpoint_fault_probe_v1",
            "status": "PASS",
            "cases": [
                {
                    "fault": fault,
                    "all_exact": True,
                    "bounded_by_save_interval": True,
                    "scientific_endpoint": True,
                    "exact": {"model": True, "optimizer": True, "rng": True},
                }
                for fault in (
                    "commit_journal_before",
                    "commit_journal_after",
                    "save_version_after_write",
                    "save_pointer_after",
                    "save_alias_after",
                    "save_journal_after",
                    "save_receipt_after",
                )
            ],
        },
        "numerical_parity": {
            "schema": EAGER_SCHEMA,
            "status": "PASS",
            "scientific_updates": 0,
            "updates_per_case_per_path": 2,
            "cases": eager_cases,
            "technical_optimizer_steps": len(eager_cases) * 8,
            "control_policy": CONTROL_POLICY,
            "control_policy_sha256": CONTROL_POLICY_SHA256,
            "runtime_admission": True,
            "producer_plan_runtime_admission": "PASS",
            "runtime_route": {
                "passed": True,
                "effective_batch_size": 32,
                "technical_optimizer_steps": 0,
                "scientific_updates": 0,
                "technical_forward_only_passes": len(FIT_IDS),
            },
        },
        "runtime_benchmark": {
            "schema": "final_refit_rgb_runtime_thread_probe_v1",
            "status": "PROFILED",
            "scientific_updates": 0,
            "best_admitted_index": 0,
            "parity_vs_first_baseline": [{"passed": True}],
            "baseline_repeat_parity": {"passed": True},
        },
    }
    technical_evidence: dict[str, dict[str, str]] = {}
    for name, payload_value in evidence_payloads.items():
        evidence_path = tmp_path / f"{name}.json"
        _write_json(evidence_path, payload_value)
        technical_evidence[name] = {
            "path": str(evidence_path),
            "sha256": sha256(evidence_path),
            "scope": "isolated CPU fixture",
        }
    admission: dict[str, Any] = {
        "schema": ADMISSION_SCHEMA,
        "status": "ADMITTED_TECHNICAL_RUNTIME_MIGRATION",
        "fit_id": "R_A5",
        "old_identity_sha256": identity["identity_sha256"],
        "parent_checkpoint_sha256": checkpoint_hash,
        "old_plan_sha256": old_plan["plan_sha256"],
        "scientific_recipe_unchanged": True,
        "optimizer_mode_unchanged": True,
        "checkpoint_interval_updates": 100,
        "scientific_optimizer_updates": 0,
        "source_sha256_before_after": {
            SOURCE_NAME: {"before_sha256": "1" * 64, "after_sha256": after_source}
        },
        "technical_evidence": technical_evidence,
    }
    admission["admission_sha256"] = canonical_sha256(admission)
    _write_json(admission_path, admission)
    new_plan_path = tmp_path / "new_plan.json"
    _plan(new_plan_path, sha256(admission_path), after_source)
    return {
        "source": source,
        "checkpoint": checkpoint,
        "old_plan": old_plan_path,
        "new_plan": new_plan_path,
        "admission": admission_path,
        "destination": tmp_path / "runtime_v2",
    }


def _migrate(paths: dict[str, Path]) -> dict[str, Any]:
    return migrate_runtime_state(
        source_directory=paths["source"],
        source_checkpoint=paths["checkpoint"],
        old_plan_path=paths["old_plan"],
        new_plan_path=paths["new_plan"],
        admission_path=paths["admission"],
        destination_run_root=paths["destination"],
        fit_id="R_A5",
    )


def test_migration_preserves_all_resumable_state_and_checkpoint_bytes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    before = paths["checkpoint"].read_bytes()
    result = _migrate(paths)
    migrated = Path(result["migrated_checkpoint"]["path"])
    assert migrated.read_bytes() == before
    assert all(result["state_equality"].values())
    assert result["completed_updates"] == 169
    assert result["next_periodic_checkpoint_update"] == 200
    assert result["scientific_optimizer_updates"] == 0
    assert (
        validate_runtime_migration(
            paths["destination"] / "fits" / "R_A5" / "RUNTIME_MIGRATION.json"
        )["migration_sha256"]
        == result["migration_sha256"]
    )


def test_binding_returns_old_identity_only_for_exact_new_execution(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _migrate(paths)
    payload = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    execution = dict(payload["identity"])
    execution.update(
        producer_contract_path=str(paths["new_plan"].resolve()),
        producer_contract_sha256=sha256(paths["new_plan"]),
        executed_source_sha256={
            SOURCE_NAME: _runtime_source_sha(),
            CONFIG_NAME: sha256(Path(CONFIG_NAME)),
        },
    )
    execution.pop("identity_sha256")
    execution["identity_sha256"] = canonical_sha256(execution)
    migration_path = paths["destination"] / "fits" / "R_A5" / "RUNTIME_MIGRATION.json"
    binding_path = paths["destination"] / "fits" / "R_A5" / "RUNTIME_RESUME_BINDING.json"
    restored = bind_runtime_resume(
        migration_path=migration_path,
        new_plan_path=paths["new_plan"],
        execution_identity=execution,
        binding_path=binding_path,
    )
    assert restored == payload["identity"]
    assert (
        validate_runtime_binding(binding_path=binding_path, execution_identity=execution)
        == payload["identity"]
    )
    assert (
        resolve_runtime_checkpoint_identity(
            run_root=paths["destination"],
            fit_id="R_A5",
            new_identity=execution,
            new_plan_path=paths["new_plan"],
        )
        == payload["identity"]
    )
    kwargs = runtime_checkpoint_binding_kwargs(run_root=paths["destination"], fit_id="R_A5")
    assert (
        kwargs["execution_binding_sha256"]
        == json.loads(binding_path.read_text(encoding="utf-8"))["binding_sha256"]
    )
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    assert kwargs["execution_binding_origin_checkpoint_sha256"] == sha256(
        Path(migration["migrated_checkpoint"]["path"])
    )

    changed_science = dict(execution)
    changed_science["seed"] = 13
    changed_science.pop("identity_sha256")
    changed_science["identity_sha256"] = canonical_sha256(changed_science)
    with pytest.raises(ValueError, match="differs from current execution"):
        validate_runtime_binding(
            binding_path=binding_path,
            execution_identity=changed_science,
        )


def test_migration_rejects_missing_periodic_curve(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["source"] / "curve_000100.json").unlink()
    with pytest.raises(ValueError, match="Missing durable periodic curve"):
        _migrate(paths)


def test_validator_rejects_rewritten_migrated_checkpoint(tmp_path: Path) -> None:
    receipt = tmp_path / "RUNTIME_MIGRATION.json"
    parent = tmp_path / "parent.pt"
    migrated = tmp_path / "migrated.pt"
    parent.write_bytes(b"same")
    migrated.write_bytes(b"different")
    value: dict[str, Any] = {
        "schema": "final_refit_runtime_migration_v1",
        "status": "MIGRATED_NOT_RESUMED",
        "scientific_optimizer_updates": 0,
        "scientific_endpoint": False,
        "checkpoint_interval_updates": 100,
        "payload_identity_preserved": True,
        "state_equality": {
            name: True
            for name in (
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "sampler_generator_state",
                "torch_rng_state",
                "cuda_rng_state_all",
                "numpy_random_state",
                "python_random_state",
                "cursor",
            )
        },
        "parent_checkpoint": {"path": str(parent), "sha256": sha256(parent)},
        "migrated_checkpoint": {"path": str(migrated), "sha256": sha256(migrated)},
    }
    value["migration_sha256"] = canonical_sha256(value)
    _write_json(receipt, value)
    with pytest.raises(ValueError, match="rewrote"):
        validate_runtime_migration(receipt)


def test_checkpoint_binding_accepts_only_legacy_origin_then_marks_future(tmp_path: Path) -> None:
    directory = tmp_path / "fits" / "R_A5"
    identity: dict[str, Any] = {"fit_id": "R_A5"}
    identity["identity_sha256"] = canonical_sha256(identity)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    generator = torch.Generator().manual_seed(7)
    cursor = {"pending_curve": [], "epoch": 1, "position": 0}

    legacy = ProducerCheckpoint(directory, identity, updates=2)
    legacy.save(model, optimizer, scheduler, generator, cursor, status="READY")
    origin_sha = sha256(directory / "checkpoint_versions" / "checkpoint_000000.pt")
    binding_sha = "a" * 64
    bound = ProducerCheckpoint(
        directory,
        identity,
        updates=2,
        execution_binding_sha256=binding_sha,
        execution_binding_origin_checkpoint_sha256=origin_sha,
    )
    assert bound.restore(model, optimizer, scheduler, generator) == cursor
    bound.begin()
    bound.commit()
    bound.save(model, optimizer, scheduler, generator, cursor, status="RUNNING")
    future = torch.load(
        directory / "checkpoint_versions" / "checkpoint_000001.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert future["execution_binding_sha256"] == binding_sha

    wrong = ProducerCheckpoint(
        directory,
        identity,
        updates=2,
        execution_binding_sha256="b" * 64,
        execution_binding_origin_checkpoint_sha256=origin_sha,
    )
    with pytest.raises(ValueError, match="execution binding differs"):
        wrong.restore(model, optimizer, scheduler, generator)
    with pytest.raises(ValueError, match="pointer execution binding differs"):
        ProducerCheckpoint(directory, identity, updates=2).restore(
            model, optimizer, scheduler, generator
        )


def test_checkpoint_restore_accepts_exact_prior_bound_origin_only(tmp_path: Path) -> None:
    directory = tmp_path / "prior_bound" / "fits" / "R_A5"
    identity: dict[str, Any] = {"fit_id": "R_A5"}
    identity["identity_sha256"] = canonical_sha256(identity)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    generator = torch.Generator().manual_seed(7)
    cursor = {"pending_curve": [], "epoch": 1, "position": 0}
    prior_sha = "c" * 64
    old = ProducerCheckpoint(
        directory,
        identity,
        updates=3,
        execution_binding_sha256=prior_sha,
        execution_binding_origin_checkpoint_sha256="d" * 64,
    )
    old.save(model, optimizer, scheduler, generator, cursor, status="RUNNING")
    origin = directory / "checkpoint_versions" / "checkpoint_000000.pt"
    origin_sha = sha256(origin)
    next_sha = "e" * 64
    migrated = ProducerCheckpoint(
        directory,
        identity,
        updates=3,
        execution_binding_sha256=next_sha,
        execution_binding_origin_checkpoint_sha256=origin_sha,
        execution_binding_origin_prior_sha256=prior_sha,
    )
    assert migrated.restore(model, optimizer, scheduler, generator) == cursor
    migrated.begin()
    migrated.commit()
    migrated.save(model, optimizer, scheduler, generator, cursor, status="RUNNING")
    future = directory / "checkpoint_versions" / "checkpoint_000001.pt"
    assert (
        torch.load(future, map_location="cpu", weights_only=False)["execution_binding_sha256"]
        == next_sha
    )

    pointer = json.loads((directory / "CHECKPOINT_POINTER.json").read_text(encoding="utf-8"))
    pointer["execution_binding_sha256"] = prior_sha
    _write_json(directory / "CHECKPOINT_POINTER.json", pointer)
    with pytest.raises(ValueError, match="pointer execution binding differs"):
        migrated.restore(model, optimizer, scheduler, generator)

    pointer["execution_binding_sha256"] = next_sha
    _write_json(directory / "CHECKPOINT_POINTER.json", pointer)
    wrong_prior = ProducerCheckpoint(
        directory,
        identity,
        updates=3,
        execution_binding_sha256="f" * 64,
        execution_binding_origin_checkpoint_sha256=origin_sha,
        execution_binding_origin_prior_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="pointer execution binding differs"):
        wrong_prior.restore(model, optimizer, scheduler, generator)


def _seal_source_evidence(
    directory: Path, payload: dict[str, Any], *, binding_sha: str | None
) -> Path:
    completed = int(payload["completed_updates"])
    checkpoint = directory / f"checkpoint_migration_origin_{completed:06d}.pt"
    if binding_sha is not None:
        payload["execution_binding_sha256"] = binding_sha
    else:
        payload.pop("execution_binding_sha256", None)
    torch.save(payload, checkpoint)
    checkpoint_hash = sha256(checkpoint)
    common = {} if binding_sha is None else {"execution_binding_sha256": binding_sha}
    _write_json(
        directory / "CHECKPOINT_POINTER.json",
        {
            "schema": "rgb_port_checkpoint_pointer_v1",
            "identity_sha256": payload["identity_sha256"],
            "completed_updates": completed,
            "version": checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
            **common,
        },
    )
    _write_json(
        directory / "CHECKPOINT_RECEIPT.json",
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "fit_id": payload["identity"]["fit_id"],
            "status": "PAUSED_RESOURCE",
            "completed_updates": completed,
            "scientific_endpoint": False,
            "identity_sha256": payload["identity_sha256"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "complete_state": True,
            "accumulation_index": 0,
            "recovery_upper": 0,
            **common,
        },
    )
    _write_json(
        directory / "UPDATE_JOURNAL.json",
        {
            "schema": "rgb_port_update_journal_v1",
            "identity_sha256": payload["identity_sha256"],
            "completed_updates": completed,
            "durable_updates": completed,
            "recovery_upper": 0,
            "pending_update_upper": 0,
            **common,
        },
    )
    return checkpoint


def _v3_fixture(tmp_path: Path) -> dict[str, Any]:
    paths = _fixture(tmp_path)
    _migrate(paths)
    original = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    execution = dict(original["identity"])
    execution.update(
        producer_contract_path=str(paths["new_plan"].resolve()),
        producer_contract_sha256=sha256(paths["new_plan"]),
        executed_source_sha256={
            SOURCE_NAME: _runtime_source_sha(),
            CONFIG_NAME: sha256(Path(CONFIG_NAME)),
        },
    )
    execution.pop("identity_sha256")
    execution["identity_sha256"] = canonical_sha256(execution)
    source_root = paths["destination"]
    a5 = source_root / "fits" / "R_A5"
    prior_binding = a5 / "RUNTIME_RESUME_BINDING.json"
    bind_runtime_resume(
        migration_path=a5 / "RUNTIME_MIGRATION.json",
        new_plan_path=paths["new_plan"],
        execution_identity=execution,
        binding_path=prior_binding,
    )
    binding_sha = json.loads(prior_binding.read_text(encoding="utf-8"))["binding_sha256"]
    a5_payload = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    a5_checkpoint = _seal_source_evidence(a5, a5_payload, binding_sha=binding_sha)

    c2f = source_root / "fits" / "R_C2F"
    c2f.mkdir()
    c2f_payload = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    c2f_identity = dict(execution)
    c2f_identity["fit_id"] = "R_C2F"
    c2f_identity.pop("identity_sha256")
    c2f_identity["identity_sha256"] = canonical_sha256(c2f_identity)
    c2f_payload["identity"] = c2f_identity
    c2f_payload["identity_sha256"] = c2f_identity["identity_sha256"]
    shutil_source = a5 / "curve_000100.json"
    (c2f / "curve_000100.json").write_bytes(shutil_source.read_bytes())
    c2f_checkpoint = _seal_source_evidence(c2f, c2f_payload, binding_sha=None)

    old_plan = json.loads(paths["new_plan"].read_text(encoding="utf-8"))
    staging_root = Path(__file__).resolve().parents[2]
    new_source_sha = sha256(staging_root / SOURCE_NAME)
    admission: dict[str, Any] = {
        "schema": OPTIMIZATION_ADMISSION_SCHEMA,
        "status": "PASS",
        "scientific_optimizer_updates": 0,
        "technical_optimizer_updates": 8,
        "technical_cases": [{"case": "default", "passed": True}],
        "admitted_options": {"resource_guard": "cached"},
        "checks": {
            "numerical_parity": "PASS",
            "state_parity": "PASS",
            "resume_parity": "PASS",
            "rng_parity": "PASS",
        },
        "old_plan_sha256": old_plan["plan_sha256"],
        "scientific_plan_sha256": canonical_sha256(
            {
                **{
                    k: v
                    for k, v in old_plan.items()
                    if k not in {"plan_sha256", "executed_source_sha256"}
                },
                "input_sha256": {
                    k: v
                    for k, v in old_plan["input_sha256"].items()
                    if k not in {"runtime_admission", "runtime_migration_admission"}
                },
            }
        ),
        "source_sha256_before_after": {
            SOURCE_NAME: {
                "before_sha256": old_plan["executed_source_sha256"][SOURCE_NAME],
                "after_sha256": new_source_sha,
            }
        },
    }
    admission["admission_sha256"] = canonical_sha256(admission)
    optimization_admission = tmp_path / "optimization_admission.json"
    _write_json(optimization_admission, admission)
    new_plan = dict(old_plan)
    new_plan["input_sha256"] = {
        **old_plan["input_sha256"],
        "runtime_optimization_admission": sha256(optimization_admission),
    }
    new_plan["executed_source_sha256"] = {SOURCE_NAME: new_source_sha}
    new_plan.pop("plan_sha256")
    new_plan["plan_sha256"] = canonical_sha256(new_plan)
    new_plan_path = tmp_path / "v3_plan.json"
    _write_json(new_plan_path, new_plan)
    return {
        "source_root": source_root,
        "old_plan": paths["new_plan"],
        "new_plan": new_plan_path,
        "admission": optimization_admission,
        "destination": tmp_path / "runtime_v3",
        "fits": {
            "R_A5": {
                "source_directory": a5,
                "source_checkpoint": a5_checkpoint,
                "curve_directory": a5,
                "prior_runtime_binding": prior_binding,
            },
            "R_C2F": {
                "source_directory": c2f,
                "source_checkpoint": c2f_checkpoint,
                "curve_directory": c2f,
                "prior_runtime_binding": None,
            },
        },
        "execution": execution,
    }


def test_v3_batch_remigrates_bound_and_unbound_trained_fits(tmp_path: Path) -> None:
    fixture = _v3_fixture(tmp_path)
    validate_optimization_admission(
        fixture["admission"],
        old_plan_path=fixture["old_plan"],
        new_plan_path=fixture["new_plan"],
    )
    result = migrate_runtime_states_v3(
        source_run_root=fixture["source_root"],
        old_plan_path=fixture["old_plan"],
        new_plan_path=fixture["new_plan"],
        optimization_admission_path=fixture["admission"],
        destination_run_root=fixture["destination"],
        fits=fixture["fits"],
    )
    assert set(result["fits"]) == {"R_A5", "R_C2F"}
    assert result["fits"]["R_A5"]["prior_runtime_sidecar_binding_sha256"] is not None
    assert result["fits"]["R_A5"]["parent_embedded_execution_binding_sha256"] is not None
    assert result["fits"]["R_C2F"]["prior_runtime_sidecar_binding_sha256"] is None
    assert result["fits"]["R_C2F"]["parent_embedded_execution_binding_sha256"] is None
    assert (
        validate_runtime_batch_migration(
            fixture["destination"] / "RUNTIME_MIGRATION_BATCH.json",
            new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
            optimization_admission_path=(
                fixture["destination"] / "RUNTIME_OPTIMIZATION_ADMISSION.json"
            ),
        )
        == result
    )
    for fit_id in ("R_A5", "R_C2F"):
        parent_payload = torch.load(
            fixture["fits"][fit_id]["source_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        execution = dict(parent_payload["identity"])
        execution.update(
            producer_contract_path=str((fixture["destination"] / "PRODUCER_PLAN.json").resolve()),
            producer_contract_sha256=sha256(fixture["destination"] / "PRODUCER_PLAN.json"),
            executed_source_sha256={
                SOURCE_NAME: _runtime_source_sha(),
                CONFIG_NAME: sha256(Path(CONFIG_NAME)),
            },
        )
        execution.pop("identity_sha256")
        execution["identity_sha256"] = canonical_sha256(execution)
        assert (
            resolve_runtime_checkpoint_identity_v3(
                run_root=fixture["destination"],
                fit_id=fit_id,
                new_identity=execution,
                new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
            )
            == parent_payload["identity"]
        )
        kwargs = runtime_checkpoint_binding_kwargs_v3(
            run_root=fixture["destination"], fit_id=fit_id
        )
        assert len(kwargs["execution_binding_sha256"]) == 64
        assert (
            kwargs.get("execution_binding_origin_prior_sha256")
            == result["fits"][fit_id]["parent_embedded_execution_binding_sha256"]
        )
        assert (
            kwargs["execution_binding_origin_checkpoint_sha256"]
            == result["fits"][fit_id]["parent_checkpoint_sha256"]
        )

    fit_directory = fixture["destination"] / "fits" / "R_A5"
    binding_sha = runtime_checkpoint_binding_kwargs_v3(
        run_root=fixture["destination"], fit_id="R_A5"
    )["execution_binding_sha256"]
    between_journal = json.loads(
        (fit_directory / "UPDATE_JOURNAL.json").read_text(encoding="utf-8")
    )
    between_journal.update(
        execution_binding_sha256=binding_sha,
        completed_updates=170,
        durable_updates=169,
        pending_update_upper=0,
    )
    _write_json(fit_directory / "UPDATE_JOURNAL.json", between_journal)
    validate_runtime_batch_migration(
        fixture["destination"] / "RUNTIME_MIGRATION_BATCH.json",
        new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
        optimization_admission_path=(
            fixture["destination"] / "RUNTIME_OPTIMIZATION_ADMISSION.json"
        ),
    )
    between_journal["completed_updates"] = 200
    _write_json(fit_directory / "UPDATE_JOURNAL.json", between_journal)
    _write_json(
        fit_directory / "curve_000200.json",
        {"updates": [{"update": update} for update in range(101, 201)]},
    )
    validate_runtime_batch_migration(
        fixture["destination"] / "RUNTIME_MIGRATION_BATCH.json",
        new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
        optimization_admission_path=(
            fixture["destination"] / "RUNTIME_OPTIMIZATION_ADMISSION.json"
        ),
    )
    _write_json(
        fit_directory / "curve_000300.json",
        {"updates": [{"update": update} for update in range(201, 301)]},
    )
    with pytest.raises(ValueError, match="advances past"):
        validate_runtime_batch_migration(
            fixture["destination"] / "RUNTIME_MIGRATION_BATCH.json",
            new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
            optimization_admission_path=(
                fixture["destination"] / "RUNTIME_OPTIMIZATION_ADMISSION.json"
            ),
        )
    (fit_directory / "curve_000300.json").unlink()
    pointer = json.loads((fit_directory / "CHECKPOINT_POINTER.json").read_text(encoding="utf-8"))
    descendant = torch.load(
        fit_directory / "checkpoint_versions" / pointer["version"],
        map_location="cpu",
        weights_only=False,
    )
    descendant["completed_updates"] = 269
    descendant["execution_binding_sha256"] = binding_sha
    descendant["cursor"]["pending_curve"] = [{"update": update} for update in range(201, 270)]
    _write_json(
        fit_directory / "curve_000200.json",
        {"updates": [{"update": update} for update in range(101, 201)]},
    )
    version = fit_directory / "checkpoint_versions" / "checkpoint_000269.pt"
    torch.save(descendant, version)
    checkpoint_hash = sha256(version)
    common = {"execution_binding_sha256": binding_sha}
    _write_json(
        fit_directory / "CHECKPOINT_POINTER.json",
        {
            "schema": "rgb_port_checkpoint_pointer_v1",
            "identity_sha256": descendant["identity_sha256"],
            "completed_updates": 269,
            "version": version.name,
            "checkpoint_sha256": checkpoint_hash,
            **common,
        },
    )
    journal = {
        "schema": "rgb_port_update_journal_v1",
        "identity_sha256": descendant["identity_sha256"],
        "completed_updates": 269,
        "durable_updates": 169,
        "recovery_upper": 0,
        "pending_update_upper": 0,
        **common,
    }
    _write_json(fit_directory / "UPDATE_JOURNAL.json", journal)
    validate_runtime_batch_migration(
        fixture["destination"] / "RUNTIME_MIGRATION_BATCH.json",
        new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
        optimization_admission_path=(
            fixture["destination"] / "RUNTIME_OPTIMIZATION_ADMISSION.json"
        ),
    )
    journal.pop("execution_binding_sha256")
    _write_json(fit_directory / "UPDATE_JOURNAL.json", journal)
    with pytest.raises(ValueError, match="recoverable descendant differs"):
        validate_runtime_batch_migration(
            fixture["destination"] / "RUNTIME_MIGRATION_BATCH.json",
            new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
            optimization_admission_path=(
                fixture["destination"] / "RUNTIME_OPTIMIZATION_ADMISSION.json"
            ),
        )
    journal["execution_binding_sha256"] = binding_sha
    _write_json(fit_directory / "UPDATE_JOURNAL.json", journal)
    active_pointer = json.loads(
        (fit_directory / "CHECKPOINT_POINTER.json").read_text(encoding="utf-8")
    )
    active_pointer["version"] = "../../../RUNTIME_MIGRATION_ORIGIN/R_A5/checkpoint_000169.pt"
    _write_json(fit_directory / "CHECKPOINT_POINTER.json", active_pointer)
    with pytest.raises(ValueError, match="path is not canonical"):
        validate_runtime_batch_migration(
            fixture["destination"] / "RUNTIME_MIGRATION_BATCH.json",
            new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
            optimization_admission_path=(
                fixture["destination"] / "RUNTIME_OPTIMIZATION_ADMISSION.json"
            ),
        )


def test_v3_rejects_tampered_prior_binding_before_publishing(tmp_path: Path) -> None:
    fixture = _v3_fixture(tmp_path)
    prior = fixture["fits"]["R_A5"]["prior_runtime_binding"]
    value = json.loads(prior.read_text(encoding="utf-8"))
    value["checkpoint_identity_sha256"] = "0" * 64
    _write_json(prior, value)
    with pytest.raises(ValueError, match="Prior runtime binding is invalid"):
        migrate_runtime_states_v3(
            source_run_root=fixture["source_root"],
            old_plan_path=fixture["old_plan"],
            new_plan_path=fixture["new_plan"],
            optimization_admission_path=fixture["admission"],
            destination_run_root=fixture["destination"],
            fits=fixture["fits"],
        )
    assert not fixture["destination"].exists()


def test_v2_lineage_distinguishes_sidecar_from_embedded_parent(tmp_path: Path) -> None:
    fixture = _v3_fixture(tmp_path)
    migrate_runtime_states_v3(
        source_run_root=fixture["source_root"],
        old_plan_path=fixture["old_plan"],
        new_plan_path=fixture["new_plan"],
        optimization_admission_path=fixture["admission"],
        destination_run_root=fixture["destination"],
        fits=fixture["fits"],
    )
    for fit_id in ("R_A5", "R_C2F"):
        source_payload = torch.load(
            fixture["fits"][fit_id]["source_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        execution = dict(source_payload["identity"])
        execution.update(
            producer_contract_path=str((fixture["destination"] / "PRODUCER_PLAN.json").resolve()),
            producer_contract_sha256=sha256(fixture["destination"] / "PRODUCER_PLAN.json"),
            executed_source_sha256={
                SOURCE_NAME: _runtime_source_sha(),
                CONFIG_NAME: sha256(Path(CONFIG_NAME)),
            },
        )
        execution.pop("identity_sha256")
        execution["identity_sha256"] = canonical_sha256(execution)
        resolve_runtime_checkpoint_identity_v3(
            run_root=fixture["destination"],
            fit_id=fit_id,
            new_identity=execution,
            new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
        )
    a5 = fixture["destination"] / "fits" / "R_A5"
    pointer = json.loads((a5 / "CHECKPOINT_POINTER.json").read_text(encoding="utf-8"))
    payload = torch.load(
        a5 / "checkpoint_versions" / pointer["version"],
        map_location="cpu",
        weights_only=False,
    )
    lineage = _validate_prior_execution_lineage(
        fit_id="R_A5",
        payload=payload,
        pointer=pointer,
        receipt=json.loads((a5 / "CHECKPOINT_RECEIPT.json").read_text(encoding="utf-8")),
        journal=json.loads((a5 / "UPDATE_JOURNAL.json").read_text(encoding="utf-8")),
        prior_binding_path=a5 / "RUNTIME_RESUME_BINDING.json",
    )
    assert lineage is not None
    assert (
        lineage["prior_runtime_sidecar_binding_sha256"]
        != lineage["parent_embedded_execution_binding_sha256"]
    )

    c2f = fixture["destination"] / "fits" / "R_C2F"
    binding = json.loads((c2f / "RUNTIME_RESUME_BINDING.json").read_text(encoding="utf-8"))
    pointer = json.loads((c2f / "CHECKPOINT_POINTER.json").read_text(encoding="utf-8"))
    payload = torch.load(
        c2f / "checkpoint_versions" / pointer["version"],
        map_location="cpu",
        weights_only=False,
    )
    receipt = json.loads((c2f / "CHECKPOINT_RECEIPT.json").read_text(encoding="utf-8"))
    journal = json.loads((c2f / "UPDATE_JOURNAL.json").read_text(encoding="utf-8"))
    for value in (payload, pointer, receipt, journal):
        value["execution_binding_sha256"] = binding["binding_sha256"]
    lineage = _validate_prior_execution_lineage(
        fit_id="R_C2F",
        payload=payload,
        pointer=pointer,
        receipt=receipt,
        journal=journal,
        prior_binding_path=c2f / "RUNTIME_RESUME_BINDING.json",
    )
    assert lineage is not None
    assert (
        lineage["prior_runtime_sidecar_binding_sha256"]
        == lineage["parent_embedded_execution_binding_sha256"]
    )


def test_v3_rejects_partial_prior_binding_evidence(tmp_path: Path) -> None:
    fixture = _v3_fixture(tmp_path)
    receipt_path = fixture["fits"]["R_A5"]["source_directory"] / "CHECKPOINT_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("execution_binding_sha256")
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="inconsistent execution bindings"):
        migrate_runtime_states_v3(
            source_run_root=fixture["source_root"],
            old_plan_path=fixture["old_plan"],
            new_plan_path=fixture["new_plan"],
            optimization_admission_path=fixture["admission"],
            destination_run_root=fixture["destination"],
            fits=fixture["fits"],
        )
    assert not fixture["destination"].exists()


def test_v3_validator_rejects_changed_destination_curve(tmp_path: Path) -> None:
    fixture = _v3_fixture(tmp_path)
    migrate_runtime_states_v3(
        source_run_root=fixture["source_root"],
        old_plan_path=fixture["old_plan"],
        new_plan_path=fixture["new_plan"],
        optimization_admission_path=fixture["admission"],
        destination_run_root=fixture["destination"],
        fits=fixture["fits"],
    )
    curve = fixture["destination"] / "fits" / "R_C2F" / "curve_000100.json"
    value = json.loads(curve.read_text(encoding="utf-8"))
    value["updates"][0]["tampered"] = True
    _write_json(curve, value)
    with pytest.raises(ValueError, match="destination evidence changed"):
        validate_runtime_batch_migration(
            fixture["destination"] / "RUNTIME_MIGRATION_BATCH.json",
            new_plan_path=fixture["destination"] / "PRODUCER_PLAN.json",
            optimization_admission_path=(
                fixture["destination"] / "RUNTIME_OPTIMIZATION_ADMISSION.json"
            ),
        )
