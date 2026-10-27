"""Recoverable RGB-only PAIR_R/R_CTX queue over admitted fold producers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json, read_json_shared


class RGBQueuePauseError(RuntimeError):
    """Clean external deadline/resource pause at an existing recovery boundary."""


def _receipt(root: Path, stage: str, status: str, **details: object) -> dict[str, Any]:
    from operational.final_refit.recipe import canonical_sha256

    value = {
        "schema": "ejepa_cloud_rgb_downstream_queue_v1",
        "stage": stage,
        "status": status,
        "rotation": int(root.name),
        "pid": os.getpid(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "event_branch_executed": False,
        "event_inputs_read": False,
        **details,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    atomic_write_json(root / "RGB_PIPELINE_RECEIPT.json", value)
    return value


def _existing_admission(root: Path, name: str) -> Path | None:
    path = root / "fits" / name / "ENDPOINT_ADMISSION.json"
    return path if path.is_file() else None


def rgb_dependency_state(run: Path, rotation: int) -> dict[str, bool]:
    """Cheap readiness view; cryptographic verification occurs before every use."""
    root = run / "rotations" / str(rotation)
    return {
        name: _existing_admission(root, name) is not None
        for name in ("R_A5", "R_C2F", "PAIR_R", "R_CTX")
    }


def _needs_context_replay(root: Path, state: Mapping[str, bool]) -> bool:
    fragment = root / "generation/R_CTX/fragment.npz"
    return not state["R_CTX"] or not (
        fragment.is_file() and fragment.with_suffix(".json").is_file()
    )


def _output_manifest(run: Path, root: Path, digest: Callable[[Path], str]) -> list[dict[str, Any]]:
    """List every durable RGB head/cache/fragment byte required for return."""
    paths = [
        root / "FOLD_TRAINING_UNIT.json",
        root / "caches/PAIR_R.npz",
        root / "caches/PAIR_R.json",
        root / "caches/R_CTX.npz",
        root / "caches/R_CTX.json",
        root / "normalizers/R_CTX.json",
    ]
    for directory in (
        root / "fits/PAIR_R",
        root / "fits/R_CTX",
        root / "refiner/R_CTX",
        root / "generation/R_CTX",
    ):
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    unique = sorted({path.resolve(strict=True) for path in paths})
    return [
        {
            "path": str(path.relative_to(run)),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in unique
    ]


def run_rgb_downstream(
    run: Path,
    rotation: int,
    *,
    deadline_utc: datetime,
    external_guard: Callable[[], None] | None = None,
    authority: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fit PAIR_R then R_CTX and emit its target-free generation fragment."""
    if rotation not in {2, 3, 4}:
        raise ValueError("Cloud RGB queue is limited to rotations 2, 3 and 4")
    if deadline_utc.tzinfo is None:
        raise ValueError("RGB queue deadline must be timezone-aware")
    from operational.final_refit import scientific_oof_pipeline as pipeline
    from operational.final_refit.fold_training import (
        build_fold_training_unit,
        write_fold_training_unit,
    )
    from operational.final_refit.recipe import sha256
    from operational.final_refit.rgb import load_all_train_source
    from operational.final_refit.scientific_authority import load_config
    from operational.final_refit.scientific_fold_heads import verify_scientific_admission

    run = run.resolve(strict=True)
    root = run / "rotations" / str(rotation)
    root.mkdir(parents=True, exist_ok=True)
    state = rgb_dependency_state(run, rotation)
    if not state["R_A5"]:
        return _receipt(root, "WAIT_R_A5", "WAITING_DEPENDENCY", dependencies=state)
    frozen_guard = pipeline.make_guard(run)

    def guard() -> None:
        if datetime.now(UTC) >= deadline_utc:
            raise RGBQueuePauseError("Authorized RGB queue deadline reached")
        frozen_guard()
        if external_guard is not None:
            external_guard()

    guard()
    verify_scientific_admission(root / "fits/R_A5/ENDPOINT_ADMISSION.json")
    config = load_config(run)
    sources = {str(path.resolve()): path for path in Path(pipeline.__file__).parent.glob("*.py")}
    unit = build_fold_training_unit(
        Path(config["fold_contract"]), rotation=rotation, executed_sources=sources
    )
    write_fold_training_unit(root / "FOLD_TRAINING_UNIT.json", unit)
    base = None
    try:
        if not state["PAIR_R"]:
            base = load_all_train_source(Path(config["rgb_source"]), Path(config["teacher"]))
            cache = pipeline.pair_cache(config, rotation, unit, base, root, "PAIR_R", guard)
            pipeline.fit_head(config, root, unit, "PAIR_R", cache, guard)
        pair_admission_path = root / "fits/PAIR_R/ENDPOINT_ADMISSION.json"
        pair_admission = verify_scientific_admission(pair_admission_path)
        _receipt(
            root,
            "PAIR_R",
            "COMPLETE",
            dependencies=rgb_dependency_state(run, rotation),
            pair_r_admission_sha256=sha256(pair_admission_path),
            pair_r_checkpoint_sha256=pair_admission["checkpoint_sha256"],
            authority=dict(authority or {}),
        )
        state = rgb_dependency_state(run, rotation)
        if not state["R_C2F"]:
            return _receipt(
                root,
                "WAIT_R_C2F",
                "WAITING_DEPENDENCY",
                dependencies=state,
                pair_r_admission_sha256=sha256(pair_admission_path),
                authority=dict(authority or {}),
            )
        guard()
        verify_scientific_admission(root / "fits/R_C2F/ENDPOINT_ADMISSION.json")
        if base is None:
            base = load_all_train_source(Path(config["rgb_source"]), Path(config["teacher"]))
        fragment = root / "generation/R_CTX/fragment.npz"
        fragment_receipt = fragment.with_suffix(".json")
        if _needs_context_replay(root, state):
            pipeline.context_branch(config, rotation, root, unit, base, "R_CTX", guard)
        context_admission_path = root / "fits/R_CTX/ENDPOINT_ADMISSION.json"
        context_admission = verify_scientific_admission(context_admission_path)
        receipt = read_json_shared(fragment_receipt)
        expected_parents = {
            name: verify_scientific_admission(
                root / "fits" / name / "ENDPOINT_ADMISSION.json"
            )["checkpoint_sha256"]
            for name in ("R_A5", "R_C2F", "PAIR_R")
        }
        if (
            receipt.get("schema") != "scientific_oof_fragment_v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("interface") != "PHASE17_RAW_RGB"
            or receipt.get("role") != "generation"
            or receipt.get("parents") != expected_parents
            or receipt.get("targets_used") is not False
            or receipt.get("output_sha256") != sha256(fragment)
        ):
            raise ValueError("RGB generation fragment differs from admitted R_CTX ancestry")
        return _receipt(
            root,
            "RGB_OOF",
            "COMPLETE",
            dependencies=rgb_dependency_state(run, rotation),
            r_ctx_admission_sha256=sha256(context_admission_path),
            r_ctx_checkpoint_sha256=context_admission["checkpoint_sha256"],
            rgb_generation_fragment=str(fragment.resolve()),
            rgb_generation_fragment_sha256=sha256(fragment),
            rgb_generation_receipt_sha256=sha256(fragment_receipt),
            parent_checkpoint_sha256=expected_parents,
            emitted_interface="PHASE17_RAW_RGB",
            phase20_context_emitted=False,
            final_ttc_predictions_emitted=False,
            artifacts=_output_manifest(run, root, sha256),
            authority=dict(authority or {}),
        )
    finally:
        if base is not None:
            base.close()


__all__ = ["RGBQueuePauseError", "rgb_dependency_state", "run_rgb_downstream"]
