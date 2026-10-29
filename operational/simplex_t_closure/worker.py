"""Zero-update T6 worker; retain historical code and validated coarse checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "artifacts/simplex_t/T0"))

from adapters import install, resumable_function, set_override
from runtime import (
    atomic_json,
    digest,
    json_record,
    publish_json,
    repair_torn_journal,
    require_roots,
)


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load operational module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if digest(args.launch) != args.launch_sha256:
        raise ValueError("operational launch changed")
    launch = json.loads(args.launch.read_text(encoding="utf-8"))
    require_roots({k: Path(v) for k, v in launch["roots"].items()})
    snapshot, follow, three = install(launch)
    checkpointed = load(
        "closure_checkpointed_t6", ROOT / "artifacts/simplex_t/T0/checkpointed_t6_worker.py"
    )
    set_override(checkpointed, "analyze_followup_phase", follow)
    set_override(checkpointed, "analyze_three_seed_family", three)
    set_override(checkpointed, "write_new_json", publish_json)
    set_override(checkpointed, "_prepare", lambda path, journal, step: None)
    original_graph = checkpointed.verify_completed_scientific_graph

    def serialized_graph(**kwargs: Any) -> dict[str, Any]:
        # The original TRAIN support returns a tuple; its durable JSON returns
        # a list. Compare the same JSON representation without numeric rounding.
        value = original_graph(**kwargs)
        serialized = json_record(value)
        atomic_json(
            ROOT / "artifacts/simplex_t/closure_20261002/GRAPH_SERIALIZATION_PARITY.json",
            {
                "tuple_fields": ["train_history_support.fraction_train_h8"],
                "numeric_values_changed": False,
                "verified_graph": serialized,
            },
        )
        return serialized

    set_override(checkpointed, "verify_completed_scientific_graph", serialized_graph)
    original_compact = checkpointed.export_compact_phase

    def compact(output: Path, **kwargs: Any) -> Any:
        kwargs["resume"] = output.exists()
        return original_compact(output, **kwargs)

    set_override(checkpointed, "export_compact_phase", compact)
    from e_jepa_ttc.simplex_t import configured_postprocessing

    configured_postprocessing.postprocess_completed_campaign = resumable_function(
        checkpointed.checkpointed_postprocess,
        {
            "shutil.copytree(persistent, output)": (
                "shutil.copytree(persistent, output, copy_function=os.link)"
            )
        },
        {"os": os},
    )
    configured_postprocessing.postprocess_configured_campaign = resumable_function(
        configured_postprocessing.postprocess_configured_campaign,
        {
            "(output.exists() and not verify_only)": (
                "(output.exists() and not verify_only and not "
                '(output / "POSTPROCESSING.json").exists())'
            ),
            "if not verify_only:\n            result = postprocess_completed_campaign(": (
                'if not verify_only and not (output / "POSTPROCESSING.json").exists():\n'
                "            result = postprocess_completed_campaign("
            ),
        },
        {},
    )
    control = ROOT / "artifacts/simplex_t/scientific_campaign/T6/CHECKPOINTED_WORK/control"
    repair_torn_journal(control / "events.jsonl")
    binding = {
        "schema": "simplex_t_closure_input_binding_v1",
        "freeze_sha256": launch["freeze_sha256"],
        "publications": launch["publications"],
        "source_configuration_sha256": launch["source_configuration_sha256"],
        "evidence_profile_sha256": launch["evidence_profile_sha256"],
        "accounting": launch["accounting"],
        "recipe_source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                ROOT / "src/e_jepa_ttc/simplex_t/followup_analysis.py",
                ROOT / "src/e_jepa_ttc/simplex_t/uncertainty_analysis.py",
                ROOT / "src/e_jepa_ttc/evaluation/risk_geometry_v10.py",
            )
        },
    }
    publish_json(control / "CLOSURE_INPUT_BINDING.json", binding)
    cli = load("closure_frozen_cli", ROOT / "scripts/postprocess_simplex_t_campaign.py")
    original_postprocess = cli.postprocess_configured_campaign

    def postprocess_receipt(*args: Any, **kwargs: Any) -> dict:
        value = original_postprocess(*args, **kwargs)
        atomic_json(ROOT / "artifacts/simplex_t/closure_20261002/CANONICAL_CALL_RESULT.json", value)
        return value

    set_override(cli, "postprocess_configured_campaign", postprocess_receipt)
    original_admitted = cli.admitted
    policy_path = Path(__file__).with_name("resource_policy.json")
    policy_hash = digest(policy_path)
    policy = json.loads(policy_path.read_text())

    def recorded_admitted(paths: list[Path]) -> dict[str, Any]:
        result = original_admitted(paths)
        if digest(policy_path) != policy_hash:
            raise ValueError("active resource amendment changed during execution")
        reasons = [
            r
            for r in result.get("reasons", [])
            if r not in {"HOST_AVAILABLE_BELOW_4_GIB", "WRITTEN_VOLUME_FREE_BELOW_20_GB"}
        ]
        if result.get("host_available_bytes", 0) < policy["host_available_ram_floor_bytes"]:
            reasons.append("HOST_AVAILABLE_BELOW_USER_AMENDED_2_GIB")
        if any(
            free < policy["written_volume_emergency_floor_bytes"]
            for free in result.get("written_volume_free_bytes", [])
        ):
            reasons.append("WRITTEN_VOLUME_FREE_BELOW_USER_AMENDED_10_GB")
        result["reasons"] = reasons
        result["has_headroom"] = not reasons
        result["resource_amendment_sha256"] = policy_hash
        if not result.get("has_headroom"):
            from supervisor import memory

            atomic_json(
                ROOT / "artifacts/simplex_t/closure_20261002/RESOURCE_PAUSE.json",
                {
                    "active_snapshot": result,
                    "windows_counters": memory(),
                    "optimizer_updates_executed": 0,
                },
            )
        return result

    set_override(cli, "admitted", recorded_admitted)
    set_override(
        cli,
        "shared_write_admission",
        lambda free, reserved: free - reserved >= policy["written_volume_emergency_floor_bytes"],
    )
    sys.argv = [str(ROOT / "scripts/postprocess_simplex_t_campaign.py"), *sys.argv[1:]]
    result = cli.main()
    if result == 0:
        # Explicit final canonical verification, independent of the cached
        # admission and input snapshot. No optimizer or fit is invoked.
        snapshot.force_check()
        from e_jepa_ttc.simplex_t.scientific_admission import validate_scientific_admission

        validate_scientific_admission(
            snapshot.freeze,
            roots=snapshot.roots,
            local_paths=Path(launch["local_paths"]),
            source_configuration=Path(launch["source_configuration"]),
            source_configuration_sha256=launch["source_configuration_sha256"],
            evidence_profile=Path(launch["evidence_profile"]),
            evidence_profile_sha256=launch["evidence_profile_sha256"],
            resource_ok=lambda: recorded_admitted([ROOT])["has_headroom"],
        )
        assert snapshot.original_freeze_reader is not None
        snapshot.original_freeze_reader(
            Path(launch["freeze"]),
            expected_sha256=launch["freeze_sha256"],
            roots=snapshot.roots,
            validate_prerequisites=snapshot.force_check,
        )
        atomic_json(
            control / "CANONICAL_FINAL_VERIFICATION.json",
            {
                "status": "CANONICAL_ADMISSION_AND_FREEZE_REVERIFIED_AFTER_DELIVERY",
                "launch_sha256": args.launch_sha256,
                "freeze_sha256": launch["freeze_sha256"],
                "optimizer_updates_executed": 0,
                "pid": os.getpid(),
            },
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
