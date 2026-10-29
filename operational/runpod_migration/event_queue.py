"""Recoverable EVENT-only PAIR_E/E_CTX queue over admitted fold producers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json, read_json_shared


class EventQueuePauseError(RuntimeError):
    """Clean external deadline/resource/input pause at a recovery boundary."""


def _receipt(root: Path, stage: str, status: str, **details: object) -> dict[str, Any]:
    from operational.final_refit.recipe import canonical_sha256

    value = {
        "schema": "ejepa_cloud_event_downstream_queue_v1",
        "stage": stage,
        "status": status,
        "rotation": int(root.name),
        "pid": os.getpid(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "rgb_model_inference_executed": False,
        **details,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    atomic_write_json(root / "EVENT_PIPELINE_RECEIPT.json", value)
    return value


def event_dependency_state(run: Path, rotation: int) -> dict[str, bool]:
    root = run / "rotations" / str(rotation) / "fits"
    return {
        name: (root / name / "ENDPOINT_ADMISSION.json").is_file()
        for name in ("E_A5", "E_C2F", "PAIR_E", "E_CTX")
    }


def _needs_context_replay(root: Path, state: Mapping[str, bool]) -> bool:
    fragment = root / "generation/E_CTX/fragment.npz"
    return not state["E_CTX"] or not (
        fragment.is_file() and fragment.with_suffix(".json").is_file()
    )


def _output_manifest(run: Path, root: Path, digest: Callable[[Path], str]) -> list[dict[str, Any]]:
    paths = [
        root / "FOLD_TRAINING_UNIT.json",
        root / "caches/PAIR_E.npz",
        root / "caches/PAIR_E.json",
        root / "caches/E_CTX.npz",
        root / "caches/E_CTX.json",
        root / "normalizers/E_CTX.json",
    ]
    for directory in (
        root / "fits/PAIR_E",
        root / "fits/E_CTX",
        root / "refiner/E_CTX",
        root / "generation/E_CTX",
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


def run_event_downstream(
    run: Path,
    rotation: int,
    *,
    deadline_utc: datetime,
    raw_gate: Callable[[], bool],
    external_guard: Callable[[], None] | None = None,
    authority: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fit PAIR_E then E_CTX and emit the target-free raw EVENT fragment."""
    if rotation not in {2, 3, 4}:
        raise ValueError("Cloud EVENT queue is limited to rotations 2, 3 and 4")
    if deadline_utc.tzinfo is None:
        raise ValueError("EVENT queue deadline must be timezone-aware")
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
    state = event_dependency_state(run, rotation)
    if not state["E_A5"]:
        return _receipt(root, "WAIT_E_A5", "WAITING_DEPENDENCY", dependencies=state)
    frozen_guard = pipeline.make_guard(run)

    def guard() -> None:
        if datetime.now(UTC) >= deadline_utc:
            raise EventQueuePauseError("Authorized EVENT queue deadline reached")
        frozen_guard()
        if external_guard is not None:
            external_guard()

    guard()
    verify_scientific_admission(root / "fits/E_A5/ENDPOINT_ADMISSION.json")
    config = load_config(run)
    sources = {str(path.resolve()): path for path in Path(pipeline.__file__).parent.glob("*.py")}
    unit = build_fold_training_unit(
        Path(config["fold_contract"]), rotation=rotation, executed_sources=sources
    )
    write_fold_training_unit(root / "FOLD_TRAINING_UNIT.json", unit)
    base = None
    try:
        base = load_all_train_source(Path(config["rgb_source"]), Path(config["teacher"]))
        if not state["PAIR_E"]:
            cache = pipeline.pair_cache(config, rotation, unit, base, root, "PAIR_E", guard)
            pipeline.fit_head(config, root, unit, "PAIR_E", cache, guard)
        pair_path = root / "fits/PAIR_E/ENDPOINT_ADMISSION.json"
        pair = verify_scientific_admission(pair_path)
        state = event_dependency_state(run, rotation)
        if not state["E_C2F"]:
            return _receipt(
                root,
                "WAIT_E_C2F",
                "WAITING_DEPENDENCY",
                dependencies=state,
                pair_e_admission_sha256=sha256(pair_path),
                pair_e_checkpoint_sha256=pair["checkpoint_sha256"],
                authority=dict(authority or {}),
            )
        guard()
        verify_scientific_admission(root / "fits/E_C2F/ENDPOINT_ADMISSION.json")
        if not raw_gate():
            return _receipt(
                root,
                "WAIT_RAW_EVENT",
                "WAITING_INPUT_COVERAGE",
                dependencies=state,
                pair_e_admission_sha256=sha256(pair_path),
                authority=dict(authority or {}),
            )
        fragment = root / "generation/E_CTX/fragment.npz"
        fragment_receipt = fragment.with_suffix(".json")
        if _needs_context_replay(root, state):
            pipeline.context_branch(config, rotation, root, unit, base, "E_CTX", guard)
        context_path = root / "fits/E_CTX/ENDPOINT_ADMISSION.json"
        context = verify_scientific_admission(context_path)
        receipt = read_json_shared(fragment_receipt)
        expected_parents = {
            name: verify_scientific_admission(
                root / "fits" / name / "ENDPOINT_ADMISSION.json"
            )["checkpoint_sha256"]
            for name in ("E_A5", "E_C2F", "PAIR_E")
        }
        if (
            receipt.get("schema") != "scientific_oof_fragment_v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("interface") != "PHASE17_RAW_EVENT"
            or receipt.get("role") != "generation"
            or receipt.get("parents") != expected_parents
            or receipt.get("targets_used") is not False
            or receipt.get("output_sha256") != sha256(fragment)
        ):
            raise ValueError("EVENT fragment differs from admitted E_CTX ancestry")
        return _receipt(
            root,
            "EVENT_OOF",
            "COMPLETE",
            dependencies=event_dependency_state(run, rotation),
            e_ctx_admission_sha256=sha256(context_path),
            e_ctx_checkpoint_sha256=context["checkpoint_sha256"],
            event_generation_fragment=str(fragment.resolve()),
            event_generation_fragment_sha256=sha256(fragment),
            event_generation_receipt_sha256=sha256(fragment_receipt),
            parent_checkpoint_sha256=expected_parents,
            emitted_interface="PHASE17_RAW_EVENT",
            phase20_context_emitted=False,
            final_ttc_predictions_emitted=False,
            artifacts=_output_manifest(run, root, sha256),
            authority=dict(authority or {}),
        )
    finally:
        if base is not None:
            base.close()


__all__ = ["EventQueuePauseError", "event_dependency_state", "run_event_downstream"]
