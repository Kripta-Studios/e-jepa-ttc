"""Linux-only preflight and explicitly assigned, resumable cloud producer fitting."""

# ruff: noqa: ANN401 -- preserve the frozen source and identity call interfaces.

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bundle import verify
from .inventory import digest, read
from .paths import PathMap


def _write_once(path: Path, value: dict[str, Any]) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != data:
        raise ValueError(f"Refusing to replace immutable migration record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def source_identity() -> dict[str, str]:
    return {path.name: digest(path) for path in Path(__file__).parent.glob("*.py")}


def prepare(bundle: Path, run: Path, budget_bytes: int) -> dict[str, Any]:
    """Verify all transferred bytes before enabling any project runtime adapter."""
    if platform.system() != "Linux":
        raise RuntimeError("This entrypoint requires Linux; local training is never a target")
    bundle, run = bundle.resolve(strict=True), run.resolve()
    if not run.is_relative_to(Path("/workspace")) or run.is_relative_to(bundle):
        raise ValueError("Cloud outputs must be separate from the immutable bundle in /workspace")
    manifest = verify(bundle)
    mapping = PathMap(manifest["roots"], str(bundle))
    original = Path(mapping.translate(manifest["source_run"])) / "SCIENTIFIC_CONFIG.json"
    if digest(original) != manifest["scientific_config_sha256"]:
        raise ValueError("Authority differs from the inventory")
    run.mkdir(parents=True, exist_ok=True)
    copied = run / "SCIENTIFIC_CONFIG.json"
    if copied.exists() and digest(copied) != digest(original):
        raise ValueError("Cloud run has a different scientific authority")
    if not copied.exists():
        shutil.copyfile(original, copied)
    migration = {
        "schema": "ejepa_linux_migration_binding_v1",
        "original_authority_sha256": digest(original),
        "bundle_manifest_sha256": digest(bundle / "BUNDLE_MANIFEST.json"),
        "runtime_sources": source_identity(),
        "bundle_root": str(bundle),
        "run_root": str(run),
        "roots": manifest["roots"],
        "pod_memory_budget_bytes": budget_bytes,
        "scientific_files_rewritten": False,
        "historical_checkpoint_import": False,
        "precision": "bf16",
        "seed": 7,
        "checkpoint_updates": 100,
        "historical_prefetch_overlay_enabled": False,
        "origin_git_commit": manifest.get("origin_git_commit"),
        "origin_git_commit_is_base_only": True,
    }
    _write_once(run / "MIGRATION_BINDING.json", migration)
    from .runtime import install

    install(mapping, budget_bytes=budget_bytes, origin_git_commit=manifest.get("origin_git_commit"))
    from operational.final_refit.scientific_authority import load_config

    config = load_config(run)
    import torch

    if platform.python_version() != "3.11.15" or str(torch.__version__) != "2.11.0+cu128":
        raise ValueError("Expected Python3.11.15 and torch2.11.0+cu128 exactly")
    if torch.version.cuda != "12.8" or not torch.cuda.is_available():
        raise ValueError("CUDA12.8 PyTorch runtime and a visible GPU are required")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Expose exactly one BF16-capable GPU per worker")
    from .resources import LinuxResourceMonitor

    allowed, resources = LinuxResourceMonitor(budget_bytes=budget_bytes).guard(
        run, checkpoint_reservation=1024**3, completed_updates=0
    )
    if not allowed:
        raise RuntimeError(f"Resource guard denied cloud preflight: {resources}")
    return {
        "manifest": manifest,
        "config": config,
        "migration": migration,
        "resources": resources,
        "gpu": torch.cuda.get_device_name(),
    }


def batch_identity(source: Any) -> dict[str, Any]:
    """Hash one deterministic, homogeneous input batch without optimizer updates."""
    import torch

    frame_count = int(source.frame_counts[0])
    indices = [i for i, count in enumerate(source.frame_counts) if count == frame_count][:4]
    batch = source.batch(indices, str(source.identity["modality"]))
    fields = {}
    for name, value in vars(batch).items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            fields[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            }
        elif isinstance(value, (list, tuple)):
            fields[name] = list(value)
    return {"indices": indices, "fields": fields, "population": source.population_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "resume-probe", "producer", "finish-rotation")
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--fit", choices=("E_A5", "E_C2F", "R_A5", "R_C2F"), default="R_A5")
    parser.add_argument("--budget-bytes", type=int, default=31_000_000_000)
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--additional-updates", type=int)
    args = parser.parse_args()
    setup = prepare(args.bundle, args.run, args.budget_bytes)
    cfg = setup["config"]
    from operational.final_refit import scientific_fold_producers as science
    from operational.final_refit.continuation_queue import queue_lock
    from operational.final_refit.fold_sources import load_fold_event_source, load_fold_rgb_source
    from operational.final_refit.recipe import canonical_sha256

    from .resources import LinuxResourceMonitor

    if args.fit.startswith("E_") and setup["manifest"]["profile"] == "rgb_pilot":
        raise ValueError("RGB pilot bundle has no event producer inputs")
    source = (
        load_fold_event_source(Path(cfg["fold_contract"]), args.rotation, Path(cfg["event_root"]))
        if args.fit.startswith("E_")
        else load_fold_rgb_source(
            Path(cfg["rgb_source"]), Path(cfg["teacher"]), Path(cfg["fold_contract"]), args.rotation
        )
    )
    try:
        recipe = science.scientific_recipe(source, args.fit, Path(cfg["config"]))
        report = {
            "schema": "ejepa_cloud_preflight_v1",
            "status": "PASS",
            "utc": datetime.now(UTC).isoformat(),
            "gpu": setup["gpu"],
            "rotation": args.rotation,
            "fit": args.fit,
            "endpoint_updates": recipe.updates,
            "population": source.population_size,
            "recipe": recipe.identity,
            "resources": setup["resources"],
            "migration_binding_sha256": digest(args.run / "MIGRATION_BINDING.json"),
            "optimizer_updates": 0,
            "linux_cuda_resume_verified": False,
        }
        report["batch_identity"] = batch_identity(source)
        modality = "RGB" if args.fit.startswith("R_") else "EVENT"
        reference_path = Path(__file__).resolve().parents[2] / (
            f"artifacts/runpod_migration_20261026/{modality}_INPUT_REFERENCE_R{args.rotation}.json"
        )
        if not reference_path.is_file():
            raise ValueError("A local CPU input reference for this rotation/modality is required")
        reference = read(reference_path)
        if (
            reference["batch_identity"] != report["batch_identity"]
            or reference["authority_sha256"] != setup["manifest"]["scientific_config_sha256"]
        ):
            raise ValueError("Transferred/decoded inputs differ from the local reference")
        report["input_parity"] = "BIT_EXACT_CPU_REFERENCE"
        report_path = args.run / f"PREFLIGHT_R{args.rotation}_{args.fit}.json"
        if args.command == "preflight":
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        if args.assignment is None:
            raise ValueError("Optimizer execution requires an explicit assignment and budget")
        assignment = read(args.assignment)
        expected_task = {
            "rotation": args.rotation,
            "fit": "FINISH_ROTATION" if args.command == "finish-rotation" else args.fit,
        }
        if (
            assignment.get("schema") != "ejepa_cloud_assignment_v1"
            or assignment.get("execution_authorized") is not True
            or expected_task not in assignment.get("tasks", [])
            or assignment.get("migration_binding_sha256") != report["migration_binding_sha256"]
            or not isinstance(assignment.get("maximum_compute_usd"), (int, float))
            or assignment["maximum_compute_usd"] <= 0
        ):
            raise ValueError("Missing explicit cloud authorization, budget, or migration binding")
        deadline = datetime.fromisoformat(assignment["execution_deadline_utc"])
        if deadline.tzinfo is None or deadline <= datetime.now(UTC):
            raise ValueError("Cloud execution deadline must be timezone-aware and in the future")
        LinuxResourceMonitor.deadline_timestamp = deadline.timestamp()
        if args.command == "resume-probe":
            from .probe import probe

            if assignment.get("technical_update_cap", 0) < 8:
                raise ValueError("Resume probe needs eight explicitly budgeted technical updates")
            with queue_lock(args.run / "CLOUD_WORKER.lock"):
                result = probe(
                    source,
                    science,
                    cfg,
                    args.run,
                    args.fit,
                    LinuxResourceMonitor(budget_bytes=args.budget_bytes),
                    report["migration_binding_sha256"],
                )
            print(json.dumps(result, indent=2))
            return 0
        exclusion_path = args.assignment.parent / "LOCAL_QUEUE_EXCLUSION.json"
        if not exclusion_path.is_file() or digest(exclusion_path) != assignment.get(
            "local_queue_exclusion_receipt_sha256"
        ):
            raise ValueError("Local queue exclusion receipt missing or changed")
        exclusion = read(exclusion_path)
        if (
            exclusion.get("status") != "EXCLUDED_FROM_LOCAL_EXECUTION"
            or expected_task not in exclusion.get("tasks", [])
            or exclusion.get("scientific_config_sha256")
            != setup["manifest"]["scientific_config_sha256"]
        ):
            raise ValueError("Local and cloud ownership have not been separated")
        if args.command == "finish-rotation":
            if setup["manifest"]["profile"] != "rotations":
                raise ValueError("Full rotation inputs are required")
            from operational.final_refit import scientific_oof_pipeline as pipeline
            from operational.final_refit.scientific_fold_heads import verify_scientific_admission
            from operational.oof_rgb_metadata_fix import repaired_rgb_metadata

            root = args.run / "rotations" / str(args.rotation)
            for producer in science.FIT_NAMES:
                admission = verify_scientific_admission(
                    root / "fits" / producer / "ENDPOINT_ADMISSION.json"
                )
                if admission["rotation"] != args.rotation:
                    raise ValueError("Producer rotation differs")
            # All four endpoints are required, so this step cannot silently train
            # an unassigned producer or import old weights under a fresh head.
            with queue_lock(args.run / "CLOUD_WORKER.lock"), repaired_rgb_metadata():
                result = pipeline.run_rotation(args.run, args.rotation)
                result["cloud_migration_sha256"] = report["migration_binding_sha256"]
                pipeline.stage_receipt(root, f"ROTATION_{args.rotation}", "COMPLETE", **result)
            print(json.dumps(result, indent=2))
            return 0
        proof_path = args.run / "technical_resume_probe" / args.fit / "RESULT.json"
        proof = read(proof_path)
        if (
            proof.get("schema") != "ejepa_linux_resume_probe_v1"
            or proof.get("technical_optimizer_updates") != 8
            or set(proof.get("cases", {})) != {"uninterrupted", "restored"}
            or proof.get("status") != "PASS"
            or proof.get("rotation") != args.rotation
            or proof.get("fit") != args.fit
            or proof.get("migration_binding_sha256") != report["migration_binding_sha256"]
            or proof.get("scientific_optimizer_updates") != 0
        ):
            raise ValueError("Matching Linux resume proof is required")
        for case in proof["cases"].values():
            if case["state"] != proof["cases"]["uninterrupted"]["state"]:
                raise ValueError("Resume proof contains differing numerical state")
        original_identity = science.identity_for

        def cloud_identity(*pos: Any, **kw: Any) -> dict[str, Any]:
            identity = original_identity(*pos, **kw)
            identity.pop("identity_sha256")
            identity.update(
                cloud_migration_sha256=report["migration_binding_sha256"],
                linux_resume_proof_sha256=digest(proof_path),
                assignment_sha256=digest(args.assignment),
            )
            identity["identity_sha256"] = canonical_sha256(identity)
            return identity

        science.identity_for = cloud_identity
        try:
            with queue_lock(args.run / "CLOUD_WORKER.lock"):
                receipt = science.train(
                    source=source,
                    fit=args.fit,
                    config=Path(cfg["config"]),
                    contract=Path(cfg["fold_contract"]),
                    run=args.run,
                    monitor=LinuxResourceMonitor(budget_bytes=args.budget_bytes),
                    device="cuda",
                    max_updates=args.additional_updates,
                )
            print(json.dumps(receipt, indent=2))
            return 0 if receipt["status"] == "COMPLETE" else 3
        finally:
            science.identity_for = original_identity
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
