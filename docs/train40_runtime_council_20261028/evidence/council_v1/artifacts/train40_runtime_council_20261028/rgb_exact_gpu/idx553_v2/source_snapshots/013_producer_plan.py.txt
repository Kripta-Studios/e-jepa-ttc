"""Standalone contract for final all-TRAIN RGB producers only.

This contract deliberately makes no claim about PAIR, context, OOF, readout,
evaluation, or submission. It lets the two deployment producers be reviewed
without freezing an unsettled upper-stage graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "final_refit_rgb_producer_plan_v1"
RGB_RUNTIME_SOURCE_MEMBERS = (
    "operational/final_refit/host_loss.py",
    "operational/final_refit/shared_resources.py",
    "operational/final_refit/optimization_probe.py",
    "operational/final_refit/eager_probe.py",
    "operational/final_refit/producers.py",
    "operational/final_refit/recipe.py",
    "operational/final_refit/rgb.py",
    "operational/final_refit/resource_monitor.py",
    "operational/final_refit/runtime_migration.py",
    "operational/rgb_port/accounting.py",
    "operational/rgb_port/recipe.py",
    "operational/rgb_port/train_producers.py",
    "operational/rgb_port_revision/loss.py",
    "src/e_jepa_ttc/data/event_v4_geometry.py",
    "src/e_jepa_ttc/data/object_event_v4.py",
    "src/e_jepa_ttc/distillation/dinov3_relational.py",
    "src/e_jepa_ttc/evaluation/garl_ttc_protocol.py",
    "src/e_jepa_ttc/evaluation/stage61_nested_pair_router.py",
    "src/e_jepa_ttc/losses/causal_scale_ttc.py",
    "src/e_jepa_ttc/models/causal_scale_ttc.py",
    "src/e_jepa_ttc/models/local_transport.py",
    "src/e_jepa_ttc/models/three_expert_router.py",
    "src/e_jepa_ttc/reproducibility.py",
    "src/e_jepa_ttc/rgb_port/contracts.py",
    "src/e_jepa_ttc/rgb_port/data.py",
    "src/e_jepa_ttc/rgb_port/features.py",
    "src/e_jepa_ttc/rgb_port/history.py",
    "src/e_jepa_ttc/rgb_port/normalization.py",
    "src/e_jepa_ttc/simplex_t/expert_phase.py",
    "src/e_jepa_ttc/training/causal_scale_eap.py",
)


def runtime_source_paths(repository: Path) -> dict[str, Path]:
    """Return every repository source executed by final RGB fitting."""
    return {name: repository / name for name in RGB_RUNTIME_SOURCE_MEMBERS}


def _sha256(path: Path) -> str:
    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProducerStage:
    """One deployment producer fixed endpoint."""

    fit_id: str
    modality: str
    base_arm: str
    seed: int
    epochs: int
    population: int
    effective_batch_size: int
    microbatch_size: int
    updates: int
    fresh_initialization: bool
    parent_checkpoints: tuple[str, ...]


@dataclass(frozen=True)
class ProducerPlan:
    """Hash-bound producer-only plan."""

    schema: str
    status: str
    scope: str
    launch_authorized: bool
    authorization_basis: str
    dataset_role: str
    queries: int
    groups: int
    stages: tuple[ProducerStage, ...]
    input_sha256: dict[str, str]
    executed_source_sha256: dict[str, str]
    upper_graph_status: str
    exclusions: tuple[str, ...]
    plan_sha256: str

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("plan_sha256")
        return value


def build_plan(
    *,
    source_descriptor: Path,
    teacher_manifest: Path,
    config: Path,
    executed_sources: dict[str, Path],
    launch_authorized: bool = False,
    runtime_admission: Path | None = None,
    runtime_migration_admission: Path | None = None,
) -> ProducerPlan:
    """Build the fixed two-producer plan without loading pixels, labels or CUDA."""
    descriptor = json.loads(source_descriptor.read_text(encoding="utf-8"))
    teacher = json.loads(teacher_manifest.read_text(encoding="utf-8"))
    if descriptor.get("schema") != "final_refit_rgb_source_descriptor_v1":
        raise ValueError("Unknown all-TRAIN RGB source descriptor")
    if teacher.get("status") != "COMPLETE_VERIFIED" or teacher.get("row_count") != 88_744:
        raise ValueError("Teacher manifest is not the complete TRAIN40 teacher")
    if not executed_sources:
        raise ValueError("Producer plan must bind every executed source")
    input_hashes = {
        "source_descriptor": _sha256(source_descriptor),
        "teacher_manifest": _sha256(teacher_manifest),
        "producer_config": _sha256(config),
        "train40_rows": str(teacher["rows_sha256"]),
    }
    if launch_authorized:
        if runtime_admission is None:
            raise ValueError("Authorized producer launch requires a runtime admission")
        admission = json.loads(runtime_admission.read_text(encoding="utf-8"))
        if (
            admission.get("schema") != "final_refit_rgb_eager_runtime_admission_v1"
            or admission.get("status") != "PASS"
            or admission.get("runtime_admission") is not True
            or admission.get("producer_plan_runtime_admission") != "PASS"
            or admission.get("scientific_updates") != 0
        ):
            raise ValueError("Runtime admission is not a zero-science PASS")
        if admission.get("source_descriptor_sha256") != _sha256(source_descriptor) or admission.get(
            "teacher_manifest_sha256"
        ) != _sha256(teacher_manifest):
            raise ValueError("Runtime admission belongs to different RGB inputs")
        admitted_sources = admission.get("executed_source_sha256")
        required_executed_names = {
            "operational/final_refit/producer_plan.py",
            *RGB_RUNTIME_SOURCE_MEMBERS,
        }
        if set(executed_sources) != required_executed_names:
            raise ValueError("Authorized producer plan lacks the exact RGB runtime source closure")
        expected_probe_sources = {
            name: _sha256(path)
            for name, path in executed_sources.items()
            if name != "operational/final_refit/producer_plan.py"
        }
        expected_probe_sources["configs/rgb_port/producers.json"] = _sha256(config)
        if not isinstance(admitted_sources, dict) or any(
            admitted_sources.get(name) != digest for name, digest in expected_probe_sources.items()
        ):
            raise ValueError("Runtime admission executed-source hashes are stale")
        bindings = descriptor.get("cache_bindings")
        if not isinstance(bindings, dict) or set(bindings) != {"P", "H", "V"}:
            raise ValueError("Authorized launch requires P/H/V RGB cache bindings")
        for role, reference in sorted(bindings.items()):
            path = (source_descriptor.resolve().parent / str(reference)).resolve(strict=True)
            input_hashes[f"rgb_cache_binding_{role}"] = _sha256(path)
        input_hashes["runtime_admission"] = _sha256(runtime_admission)
        if runtime_migration_admission is not None:
            migration = json.loads(runtime_migration_admission.read_text(encoding="utf-8"))
            if (
                migration.get("schema") != "final_refit_runtime_migration_admission_v1"
                or migration.get("status") != "ADMITTED_TECHNICAL_RUNTIME_MIGRATION"
                or migration.get("scientific_optimizer_updates") != 0
                or migration.get("scientific_recipe_unchanged") is not True
                or migration.get("checkpoint_interval_updates") != 100
            ):
                raise ValueError("Invalid separate runtime migration admission")
            unsigned = dict(migration)
            claimed = unsigned.pop("admission_sha256", None)
            if claimed != _canonical(unsigned):
                raise ValueError("Runtime migration admission hash differs")
            input_hashes["runtime_migration_admission"] = _sha256(runtime_migration_admission)
    updates = 18 * ((88_744 + 31) // 32)
    stages = tuple(
        ProducerStage(
            fit_id=fit_id,
            modality="rgb",
            base_arm=base_arm,
            seed=7,
            epochs=18,
            population=88_744,
            effective_batch_size=32,
            microbatch_size=32,
            updates=updates,
            fresh_initialization=True,
            parent_checkpoints=(),
        )
        for fit_id, base_arm in (("R_A5", "A5"), ("R_C2F", "C2F"))
    )
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "FROZEN_AUTHORIZED_PRODUCER_SCOPE"
            if launch_authorized
            else "PREPARED_PENDING_CACHE_BINDINGS_AND_RUNTIME_ADMISSION"
        ),
        "scope": "fresh final all-TRAIN40 RGB producers only",
        "launch_authorized": launch_authorized,
        "authorization_basis": (
            "explicit user instruction to prepare and train the final TRAIN40 model"
        ),
        "dataset_role": "TRAIN",
        "queries": 88_744,
        "groups": 40,
        "stages": [asdict(item) for item in stages],
        "input_sha256": input_hashes,
        "executed_source_sha256": {
            name: _sha256(path) for name, path in sorted(executed_sources.items())
        },
        "upper_graph_status": "UNDECIDED_OOF_REVIEW_PENDING",
        "exclusions": [
            "PAIR_E",
            "PAIR_R",
            "E_CTX",
            "R_CTX",
            "PHASE_ONLY",
            "OOF rotations",
            "evaluation",
            "test12 inference",
            "CodaBench submission",
        ],
    }
    return ProducerPlan(
        **{**value, "stages": stages, "exclusions": tuple(value["exclusions"])},
        plan_sha256=_canonical(value),
    )


def write_plan(path: Path, plan: ProducerPlan) -> None:
    """Atomically create an immutable producer plan."""
    encoded = json.dumps(asdict(plan), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("Refusing to replace a changed producer plan")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_plan(path: Path) -> ProducerPlan:
    """Load and validate the producer-only graph and identity."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("Unknown producer plan schema")
    stages = tuple(
        ProducerStage(**{**item, "parent_checkpoints": tuple(item["parent_checkpoints"])})
        for item in value.pop("stages")
    )
    exclusions = tuple(value.pop("exclusions"))
    plan = ProducerPlan(stages=stages, exclusions=exclusions, **value)
    if (
        _canonical(plan.payload()) != plan.plan_sha256
        or plan.status
        not in {
            "PREPARED_PENDING_CACHE_BINDINGS_AND_RUNTIME_ADMISSION",
            "FROZEN_AUTHORIZED_PRODUCER_SCOPE",
        }
        or plan.launch_authorized != (plan.status == "FROZEN_AUTHORIZED_PRODUCER_SCOPE")
        or plan.queries != 88_744
        or plan.groups != 40
        or [(item.fit_id, item.updates) for item in plan.stages]
        != [("R_A5", 49_932), ("R_C2F", 49_932)]
        or plan.upper_graph_status != "UNDECIDED_OOF_REVIEW_PENDING"
    ):
        raise ValueError("Producer plan identity or fixed endpoint differs")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze the two-leaf producer scope without admitting optimizer updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-descriptor", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-scientific-launch", action="store_true")
    parser.add_argument("--runtime-admission", type=Path)
    parser.add_argument("--runtime-migration-admission", type=Path)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    plan = build_plan(
        source_descriptor=args.source_descriptor.resolve(strict=True),
        teacher_manifest=args.teacher_manifest.resolve(strict=True),
        config=args.config.resolve(strict=True),
        executed_sources={
            "operational/final_refit/producer_plan.py": Path(__file__).resolve(),
            **runtime_source_paths(repository),
        },
        launch_authorized=args.authorize_scientific_launch,
        runtime_admission=(
            args.runtime_admission.resolve(strict=True) if args.runtime_admission else None
        ),
        runtime_migration_admission=(
            args.runtime_migration_admission.resolve(strict=True)
            if args.runtime_migration_admission
            else None
        ),
    )
    write_plan(args.output.resolve(), plan)
    print(json.dumps(asdict(plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RGB_RUNTIME_SOURCE_MEMBERS",
    "ProducerPlan",
    "ProducerStage",
    "build_plan",
    "load_plan",
    "main",
    "runtime_source_paths",
    "write_plan",
]
