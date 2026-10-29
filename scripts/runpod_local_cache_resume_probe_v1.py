"""Eight-update technical resume proof for local-cache and bounded-prefetch routing."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json
from operational.runpod_migration.inventory import digest, read
from operational.runpod_migration.local_cache import prove_rgb_route
from operational.runpod_migration.runtime_prefetch import ByteBoundedFoldPrefetchRuntime


def _validate_assignment(value: dict[str, Any], migration_sha256: str, rotation: int) -> float:
    task = {"rotation": rotation, "fit": "R_A5"}
    if (
        value.get("schema") != "ejepa_cloud_assignment_v1"
        or value.get("execution_authorized") is not True
        or task not in value.get("tasks", [])
        or value.get("migration_binding_sha256") != migration_sha256
        or int(value.get("technical_update_cap", 0)) < 8
    ):
        raise ValueError("Local-cache resume proof lacks an eight-update technical assignment")
    deadline = datetime.fromisoformat(str(value["execution_deadline_utc"]))
    if deadline.tzinfo is None or deadline <= datetime.now(UTC):
        raise ValueError("Technical assignment deadline is invalid")
    return deadline.timestamp()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=range(5), required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, default=31_000_000_000)
    args = parser.parse_args()

    import runpod_technical_overlay_v3 as technical

    setup = technical.prepare(args.bundle, args.run, args.budget_bytes)
    migration_sha256 = digest(args.run / "MIGRATION_BINDING.json")
    deadline = _validate_assignment(read(args.assignment), migration_sha256, args.rotation)
    from operational.final_refit import scientific_fold_producers as science
    from operational.final_refit.fold_sources import load_fold_rgb_source
    from operational.runpod_migration import resources
    from operational.runpod_migration.probe import probe

    resources.LinuxResourceMonitor.deadline_timestamp = deadline
    config = setup["config"]
    source = load_fold_rgb_source(
        Path(config["rgb_source"]),
        Path(config["teacher"]),
        Path(config["fold_contract"]),
        args.rotation,
    )
    original_train = science.train

    def bounded_train(*pos: Any, **kw: Any) -> dict[str, Any]:
        with ByteBoundedFoldPrefetchRuntime(science):
            return original_train(*pos, **kw)

    try:
        input_proof = prove_rgb_route(
            source,
            logical_root=args.bundle,
            stage_root=args.stage,
            readiness_path=args.readiness,
        )
        science.train = bounded_train
        resume = probe(
            source,
            science,
            config,
            args.run,
            "R_A5",
            resources.LinuxResourceMonitor(budget_bytes=args.budget_bytes),
            migration_sha256,
        )
    finally:
        science.train = original_train
        source.close()
    result = {
        "schema": "ejepa_local_cache_resume_probe_v1",
        "status": "PASS" if input_proof["status"] == resume["status"] == "PASS" else "FAIL",
        "rotation": args.rotation,
        "fit": "R_A5",
        "input_proof": input_proof,
        "resume_probe": resume,
        "migration_binding_sha256": migration_sha256,
        "readiness_sha256": digest(args.readiness),
        "local_cache_source_sha256": digest(
            Path(__file__).resolve().parents[1] / "operational/runpod_migration/local_cache.py"
        ),
        "runtime_prefetch_source_sha256": digest(
            Path(__file__).resolve().parents[1] / "operational/runpod_migration/runtime_prefetch.py"
        ),
        "bounded_prefetch_source_sha256": digest(
            Path(__file__).resolve().parents[1] / "operational/runpod_migration/bounded_prefetch.py"
        ),
        "technical_optimizer_updates": resume["technical_optimizer_updates"],
        "scientific_optimizer_updates": 0,
        "scientific_execution_allowed": False,
    }
    from operational.final_refit.recipe import canonical_sha256

    result["proof_sha256"] = canonical_sha256(result)
    atomic_write_json(args.run / "LOCAL_CACHE_RESUME_PROOF.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
