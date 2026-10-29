"""Explicitly admitted R2/R3/R4 R_C2F cloud producers; frozen science is unchanged."""

# ruff: noqa: ANN401

from __future__ import annotations

import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__:
    from . import runpod_resources_v3 as resources
    from . import runpod_technical_overlay_v3 as technical
else:
    import runpod_resources_v3 as resources
    import runpod_technical_overlay_v3 as technical

QUOTA_TTL_S = 120.0
QUOTA_REFRESH_S = 30.0
QUOTA_INFLIGHT_RESERVE = 4 * 1024**3
ROTATIONS = {2, 3, 4}
FIT = "R_C2F"
C2F_MIGRATION_BUDGET_BYTES = 60_999_999_488


class QuotaPublisher:
    """Keep expensive volume inventory outside each optimizer critical path."""

    def __init__(self, root: Path, initial_free: int, initial: dict[str, Any]) -> None:
        self.root = root
        self.original = resources.storage_available
        self.snapshot: tuple[float, int, dict[str, Any], str | None] = (
            time.monotonic(),
            initial_free,
            initial,
            None,
        )
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.publish, daemon=True)

    def publish(self) -> None:
        while not self.stop.wait(QUOTA_REFRESH_S):
            try:
                free, report = self.original(self.root)
                self.snapshot = (time.monotonic(), free, report, None)
            except Exception as error:
                self.snapshot = (time.monotonic(), 0, {}, str(error))

    def read(self, root: Path) -> tuple[int, dict[str, Any]]:
        if root.resolve() != self.root.resolve():
            raise ValueError("Quota snapshot belongs to another scientific run")
        observed, free, report, error = self.snapshot
        age = time.monotonic() - observed
        if error is not None or not 0 <= age <= QUOTA_TTL_S:
            raise RuntimeError(f"Cloud quota inventory unavailable/stale: age={age}, error={error}")
        return max(0, free - QUOTA_INFLIGHT_RESERVE), {
            **report,
            "snapshot_age_s": age,
            "snapshot_ttl_s": QUOTA_TTL_S,
            "extra_inflight_reserve_bytes": QUOTA_INFLIGHT_RESERVE,
            "inventory_off_optimizer_critical_path": True,
        }


def validate_budget(assignment: dict[str, Any], budget: dict[str, Any], rotation: int) -> None:
    """Admit exactly one C2F producer without expanding the shared user cap."""
    task = [{"rotation": rotation, "fit": FIT}]
    limit, assigned = budget.get("maximum_total_usd"), assignment.get("maximum_compute_usd")
    if (
        rotation not in ROTATIONS
        or budget.get("schema") != "ejepa_cloud_budget_authorization_v1"
        or not isinstance(limit, (int, float))
        or not 0 < limit <= 24
        or not isinstance(assigned, (int, float))
        or not 0 < assigned <= limit
        or assignment.get("schema") != "ejepa_cloud_assignment_v1"
        or assignment.get("execution_authorized") is not True
        or assignment.get("tasks") != task
    ):
        raise ValueError("C2F assignment exceeds or differs from the authorized cloud scope")
    end = datetime.fromisoformat(str(assignment["execution_deadline_utc"]))
    cap = datetime.fromisoformat(str(budget["execution_deadline_utc"]))
    if end.tzinfo is None or cap.tzinfo is None or not datetime.now(UTC) < end <= cap:
        raise ValueError("C2F assignment deadline is invalid or exceeds total budget window")


def validate_runtime_budget(budget_bytes: int) -> None:
    """Bind every new C2F migration to the exact discovered shared-cgroup limit."""
    if budget_bytes != C2F_MIGRATION_BUDGET_BYTES:
        raise ValueError("New C2F runs require the exact discovered cgroup budget")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "producer":
        raise ValueError("C2F overlay supports only scientific producer execution")
    from operational.runpod_migration import runner
    from operational.runpod_migration.inventory import digest, read

    def argument(name: str) -> str:
        if sys.argv.count(name) != 1:
            raise ValueError(f"Exactly one {name} is required")
        index = sys.argv.index(name)
        if index + 1 >= len(sys.argv):
            raise ValueError(f"Missing value for {name}")
        return sys.argv[index + 1]

    rotation = int(argument("--rotation"))
    if rotation not in ROTATIONS or argument("--fit") != FIT:
        raise ValueError("Only explicitly assigned R2/R3/R4 R_C2F tasks are admitted")
    assignment_path = Path(argument("--assignment"))
    assignment = read(assignment_path)
    budget_path = assignment_path.parent / "USER_BUDGET_24USD.json"
    validate_budget(assignment, read(budget_path), rotation)
    publisher: QuotaPublisher | None = None

    def prepare(bundle: Path, run: Path, budget_bytes: int) -> dict[str, Any]:
        nonlocal publisher
        validate_runtime_budget(budget_bytes)
        setup = technical.prepare(bundle, run, budget_bytes)
        proof_path = run / f"technical_resume_probe/{FIT}/RESULT.json"
        proof = read(proof_path)
        exclusion_path = assignment_path.parent / "LOCAL_QUEUE_EXCLUSION.json"
        exclusion = read(exclusion_path)
        compatibility_path = assignment_path.parent / "WINDOWS_CHECKPOINT_COMPATIBILITY.json"
        compatibility = read(compatibility_path)
        input_reference = (
            Path(runner.__file__).resolve().parents[2]
            / f"artifacts/runpod_migration_20261026/RGB_INPUT_REFERENCE_R{rotation}.json"
        )
        if (
            proof.get("schema") != "ejepa_linux_resume_probe_v1"
            or proof.get("status") != "PASS"
            or proof.get("rotation") != rotation
            or proof.get("fit") != FIT
            or proof.get("scientific_optimizer_updates") != 0
            or proof.get("technical_optimizer_updates") != 8
            or set(proof.get("cases", {})) != {"uninterrupted", "restored"}
            or any(
                case.get("state") != proof["cases"]["uninterrupted"].get("state")
                for case in proof["cases"].values()
            )
            or exclusion.get("schema") != "ejepa_cloud_local_exclusion_c2f_v1"
            or exclusion.get("status") != "EXCLUDED_FROM_LOCAL_EXECUTION"
            or exclusion.get("tasks") != [{"rotation": rotation, "fit": FIT}]
            or digest(exclusion_path) != assignment.get("local_queue_exclusion_receipt_sha256")
            or compatibility.get("status") != "PASS"
            or compatibility.get("resume_proof_sha256") != digest(proof_path)
            or compatibility.get("cross_hardware_numerical_continuity_claimed") is not False
            or not input_reference.is_file()
        ):
            raise ValueError("C2F launch lacks matching input/resume/ownership/Windows evidence")
        admission = {
            "schema": "ejepa_scientific_cloud_admission_c2f_v1",
            "status": "PASS",
            "task": {"rotation": rotation, "fit": FIT},
            "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
            "scientific_entrypoint_sha256": digest(Path(__file__)),
            "assignment_sha256": digest(assignment_path),
            "budget_sha256": digest(budget_path),
            "local_queue_exclusion_sha256": digest(exclusion_path),
            "resume_proof_sha256": digest(proof_path),
            "windows_compatibility_sha256": digest(compatibility_path),
            "input_reference_sha256": digest(input_reference),
            "scientific_execution_allowed": True,
            "fit_specific_proof_reused_from_a5": False,
            "frozen_c2f_recipe_selected_by_runner": True,
            "quota_policy": {
                "ttl_seconds": QUOTA_TTL_S,
                "refresh_seconds": QUOTA_REFRESH_S,
                "inflight_reserve_bytes": QUOTA_INFLIGHT_RESERVE,
            },
            "checkpoint_interval_updates": 100,
            "shared_cgroup_memory_guard": True,
            "migration_bound_memory_budget_bytes": budget_bytes,
            "dynamic_memory_guard_is_external_operational_evidence": True,
            "physical_memory_reserve_bytes": 2 * 1024**3,
            "billing_stop_is_not_implied_by_python_deadline": True,
        }
        admission_path = run / "SCIENTIFIC_CLOUD_ADMISSION.json"
        runner._write_once(admission_path, admission)
        from operational.final_refit import scientific_fold_producers as science
        from operational.final_refit.recipe import canonical_sha256

        original_identity = science.identity_for

        def identity(*args: Any, **kwargs: Any) -> dict[str, Any]:
            value = original_identity(*args, **kwargs)
            value.pop("identity_sha256")
            value["scientific_cloud_admission_sha256"] = digest(admission_path)
            value["identity_sha256"] = canonical_sha256(value)
            return value

        science.identity_for = identity
        publisher = QuotaPublisher(
            run, setup["resources"]["disk_free_bytes"], setup["resources"]["storage_allocation"]
        )
        resources.storage_available = publisher.read
        publisher.thread.start()
        return setup

    runner.prepare = prepare
    try:
        return runner.main()
    finally:
        if publisher is not None:
            publisher.stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
