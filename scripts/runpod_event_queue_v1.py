"""Run only PAIR_E and E_CTX after verified event producers and raw inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path, PureWindowsPath
from typing import Any

from operational.rgb_port.accounting import atomic_write_json
from operational.runpod_migration.event_queue import EventQueuePauseError, run_event_downstream
from operational.runpod_migration.inventory import digest, read

TASK = "EVENT_DOWNSTREAM"
EVENT_MEMORY_BUDGET_BYTES = 60_999_999_488


class _QueueDirectoryMap:
    """Extend the verified supplemental map to admitted directory roots."""

    def __init__(self, mapping: Any) -> None:
        self.mapping = mapping

    def translate(self, value: str | PathLike[str]) -> str:
        windows = PureWindowsPath(str(value))
        if ".." in windows.parts:
            raise ValueError("Unresolved traversal in EVENT queue path")
        if windows.drive:
            try:
                suffix = windows.relative_to(PureWindowsPath("E:/eAP_dataset"))
            except ValueError:
                pass
            else:
                return str(Path("/workspace/ejepa_event_raw", *suffix.parts))
        translated = self.mapping.translate(value)
        if Path(translated).exists():
            return translated
        primary = getattr(self.mapping, "primary", None)
        roots: Sequence[Path] = getattr(self.mapping, "roots", ())
        if primary is None:
            return translated
        try:
            relative = primary.relative(value)
        except ValueError:
            return translated
        for root in roots:
            candidate = root / relative
            if candidate.exists():
                return str(candidate)
        return translated


def _install_rgb_routing(
    run: Path, config: dict[str, Any], verified_mapping: Any
) -> None:
    """Resolve descriptor-relative historical paths through admitted roots."""
    from operational.final_refit import rgb

    original = json.loads((run / "SCIENTIFIC_CONFIG.json").read_text(encoding="utf-8"))
    original_base = PureWindowsPath(str(original["rgb_source"])).parent
    original_loader = rgb.load_all_train_source

    def resolve(raw: str) -> Path:
        candidate = PureWindowsPath(raw)
        source = candidate if candidate.is_absolute() else original_base / candidate
        normalized = ntpath.normpath(str(source))
        return Path(verified_mapping.translate(normalized)).resolve(strict=True)

    def routed_loader(
        descriptor_path: Path, teacher_manifest: Path | None = None
    ) -> Any:
        descriptor_path = Path(descriptor_path).resolve(strict=True)
        value = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if value.get("schema") != "final_refit_rgb_source_descriptor_v1":
            raise ValueError("Unknown final-refit RGB source descriptor")
        roles = {
            role: resolve(str(value["role_manifests"][role])) for role in rgb.ROLES
        }
        caches = {
            role: resolve(str(path))
            for role, path in value.get("cache_bindings", {}).items()
        }
        teacher = (
            Path(teacher_manifest).resolve(strict=True)
            if teacher_manifest is not None
            else resolve(str(value["teacher_manifest"]))
        )
        return rgb.AllTrainRGBSource(roles, teacher, cache_bindings=caches)

    rgb.load_all_train_source = routed_loader
    for name, module in tuple(sys.modules.items()):
        if (
            name.startswith("operational.final_refit.")
            and vars(module).get("load_all_train_source") is original_loader
        ):
            vars(module)["load_all_train_source"] = routed_loader


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"Refusing to replace immutable EVENT record: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")


def _validate_authority(
    assignment_path: Path, budget_path: Path, exclusion_path: Path, run: Path, rotation: int
) -> tuple[datetime, dict[str, str]]:
    assignment, budget, exclusion = map(read, (assignment_path, budget_path, exclusion_path))
    binding = run / "EVENT_MIGRATION_BINDING.json"
    task = [{"rotation": rotation, "fit": TASK}]
    deadline = datetime.fromisoformat(str(assignment.get("execution_deadline_utc")))
    cap = datetime.fromisoformat(str(budget.get("execution_deadline_utc")))
    assigned, total = assignment.get("maximum_compute_usd"), budget.get("maximum_total_usd")
    if (
        assignment.get("schema") != "ejepa_cloud_assignment_v1"
        or assignment.get("execution_authorized") is not True
        or assignment.get("tasks") != task
        or assignment.get("migration_binding_sha256") != digest(binding)
        or budget.get("schema") != "ejepa_event_cloud_budget_authorization_v1"
        or budget.get("execution_authorized") is not True
        or not isinstance(assigned, (int, float))
        or not isinstance(total, (int, float))
        or not 0 < assigned <= total <= 24
        or deadline.tzinfo is None
        or cap.tzinfo is None
        or not datetime.now(UTC) < deadline <= cap
    ):
        raise ValueError("EVENT downstream assignment, budget, binding or deadline is invalid")
    if (
        exclusion.get("schema") != "ejepa_event_downstream_local_exclusion_v1"
        or exclusion.get("status") != "EXCLUDED_FROM_LOCAL_EXECUTION"
        or exclusion.get("tasks") != task
        or digest(exclusion_path) != assignment.get("local_queue_exclusion_receipt_sha256")
    ):
        raise ValueError("EVENT downstream local/cloud ownership is absent or changed")
    return deadline, {
        "assignment_sha256": digest(assignment_path),
        "budget_sha256": digest(budget_path),
        "exclusion_sha256": digest(exclusion_path),
        "event_migration_binding_sha256": digest(binding),
        "queue_entrypoint_sha256": digest(Path(__file__)),
        "queue_library_sha256": digest(
            Path(__file__).resolve().parents[1]
            / "operational/runpod_migration/event_queue.py"
        ),
    }


def _verify_metadata_closure(config: dict[str, Any], rotation: int) -> dict[str, Any]:
    """Exercise strict metadata loaders without decoding RGB or running a model."""
    from operational.final_refit.rgb import load_all_train_source
    from operational.final_refit.scientific_oof_features import role_rows

    base = load_all_train_source(Path(config["rgb_source"]), Path(config["teacher"]))
    try:
        roles = {}
        for role in ("producer", "refiner", "generation"):
            rows = role_rows(Path(config["fold_contract"]), rotation, role)
            sample = base.row(int(rows.iloc[0].source_ordinal))
            if (
                str(sample["sample_token"]) != str(rows.iloc[0].sample_token)
                or str(sample["group_id"]) != str(rows.iloc[0].group_id)
            ):
                raise ValueError("EVENT target metadata differs from fold role identity")
            roles[role] = {
                "rows": len(rows),
                "groups": int(rows.group_id.astype(str).nunique()),
                "ordered_token_sha256": _canonical(rows.sample_token.astype(str).tolist()),
            }
        return {
            "schema": "ejepa_event_downstream_metadata_closure_v1",
            "status": "PASS",
            "rotation": rotation,
            "roles": roles,
            "population": base.population_size,
            "rgb_pixels_decoded": False,
            "rgb_model_inference_executed": False,
            "target_metadata_rows_available": True,
        }
    finally:
        base.close()


def _verify_supplemental(path: Path, authority_sha256: str) -> dict[str, str]:
    receipt = read(path)
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != "ejepa_event_downstream_supplemental_admission_v1"
        or receipt.get("status") != "VERIFIED"
        or receipt.get("scientific_config_sha256") != authority_sha256
        or receipt.get("files") != 13
        or receipt.get("total_bytes") != 81_182_038
        or receipt.get("hashes_verified") != 13
        or receipt.get("mapping_verified") is not True
        or not isinstance(receipt.get("manifest_sha256"), str)
        or not isinstance(receipt.get("immutable_seal_sha256"), str)
        or receipt.get("receipt_sha256") != _canonical(payload)
    ):
        raise ValueError("EVENT downstream supplemental closure is absent or changed")
    return {"event_supplemental_admission_sha256": digest(path)}


def _verify_producer_cloud(run: Path, rotation: int, fit: str) -> None:
    endpoint = run / "rotations" / str(rotation) / "fits" / fit
    if not (endpoint / "ENDPOINT_ADMISSION.json").is_file():
        return
    from operational.final_refit.scientific_fold_heads import verify_scientific_admission

    verify_scientific_admission(endpoint / "ENDPOINT_ADMISSION.json")
    cloud = read(endpoint / "SCIENTIFIC_CLOUD_ADMISSION.json")
    if (
        cloud.get("schema") != "ejepa_event_scientific_cloud_admission_v1"
        or cloud.get("status") != "ADMITTED"
        or cloud.get("rotation") != rotation
        or cloud.get("fit") != fit
        or not isinstance(cloud.get("assignment_sha256"), str)
        or not isinstance(cloud.get("probe_result_sha256"), str)
        or not isinstance(cloud.get("input_identity_sha256"), str)
    ):
        raise ValueError(f"{fit} lacks its exact EVENT cloud admission")


def _raw_gate(path: Path, expected_manifest_sha: str) -> bool:
    if not path.is_file():
        return False
    receipt = read(path)
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") == "ejepa_event_raw_coverage_v1"
        and receipt.get("status") in {"TRANSFERRING", "HASHING"}
    ):
        return False
    if (
        receipt.get("schema") != "ejepa_event_raw_coverage_v1"
        or receipt.get("status") != "VERIFIED"
        or receipt.get("raw_deferred_manifest_sha256") != expected_manifest_sha
        or receipt.get("files") != 41
        or receipt.get("sequences") != 40
        or receipt.get("total_bytes") != 449_003_803_674
        or receipt.get("hashes_verified") != 41
        or receipt.get("availability_verified") is not True
        or receipt.get("receipt_sha256") != _canonical(payload)
    ):
        raise ValueError("Deferred raw EVENT coverage receipt is invalid or changed")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--exclusion", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--raw-coverage", type=Path, required=True)
    parser.add_argument("--supplemental-admission", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("Poll interval must be positive")
    from operational.runpod_migration.event_runner import _mapping, prepare_event
    from operational.runpod_migration.paths import mapped_path_class

    setup = prepare_event(args.bundle, args.run, EVENT_MEMORY_BUDGET_BYTES)
    # The import hook covers modules loaded after preparation. Repair only the
    # same Path alias in final_refit modules imported transitively beforehand.
    verified_mapping = _mapping(setup["manifest"], args.bundle.resolve(strict=True))
    path_type = mapped_path_class(_QueueDirectoryMap(verified_mapping))
    _install_rgb_routing(args.run.resolve(strict=True), setup["config"], verified_mapping)
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("operational.final_refit.", "e_jepa_ttc.")) and "Path" in vars(
            module
        ):
            vars(module)["Path"] = path_type
    from operational.final_refit.continuation_queue import queue_lock
    from operational.final_refit.scientific_oof_pipeline import ResourcePause

    run = args.run.resolve(strict=True)
    deadline, authority = _validate_authority(
        args.assignment.resolve(strict=True),
        args.budget.resolve(strict=True),
        args.exclusion.resolve(strict=True),
        run,
        args.rotation,
    )
    raw_manifest_sha = digest(args.raw_manifest.resolve(strict=True))
    authority.update(
        _verify_supplemental(
            args.supplemental_admission.resolve(strict=True),
            setup["migration"]["original_authority_sha256"],
        )
    )
    metadata = _verify_metadata_closure(setup["config"], args.rotation)
    metadata["receipt_sha256"] = _canonical(metadata)
    metadata_path = run / f"EVENT_DOWNSTREAM_METADATA_R{args.rotation}.json"
    _write_once(metadata_path, metadata)
    authority["metadata_closure_sha256"] = digest(metadata_path)
    authority["raw_deferred_manifest_sha256"] = raw_manifest_sha
    admission = {
        "schema": "ejepa_event_downstream_admission_v1",
        "status": "PASS",
        "task": {"rotation": args.rotation, "fit": TASK},
        **authority,
        "allowed_new_endpoints": ["PAIR_E", "E_CTX"],
        "allowed_generation_interface": "PHASE17_RAW_EVENT",
        "rgb_model_inference_authorized": False,
        "scientific_pipeline_edited": False,
    }
    admission_path = run / f"EVENT_DOWNSTREAM_ADMISSION_R{args.rotation}.json"
    _write_once(admission_path, admission)
    authority["event_downstream_admission_sha256"] = digest(admission_path)
    while datetime.now(UTC) < deadline:
        try:
            with queue_lock(run / f"EVENT_DOWNSTREAM_R{args.rotation}.lock"):
                _verify_producer_cloud(run, args.rotation, "E_A5")
                _verify_producer_cloud(run, args.rotation, "E_C2F")
                result = run_event_downstream(
                    run,
                    args.rotation,
                    deadline_utc=deadline,
                    raw_gate=lambda: _raw_gate(args.raw_coverage, raw_manifest_sha),
                    authority=authority,
                )
        except (EventQueuePauseError, ResourcePause) as error:
            atomic_write_json(
                run / f"EVENT_DOWNSTREAM_PAUSE_R{args.rotation}.json",
                {
                    "schema": "ejepa_event_downstream_pause_v1",
                    "status": "PAUSED_RECOVERABLE",
                    "rotation": args.rotation,
                    "reason": str(error),
                    "utc": datetime.now(UTC).isoformat(),
                },
            )
            return 4
        if result["status"] == "COMPLETE" and result["stage"] == "EVENT_OOF":
            _write_once(run / f"EVENT_DOWNSTREAM_RESULT_R{args.rotation}.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.once:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 3
        time.sleep(min(args.poll_seconds, max(0.0, deadline.timestamp() - time.time())))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
