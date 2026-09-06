"""Verify discovered Stage70 interfaces and historical expert bindings read-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest
from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json


def main() -> None:
    """Audit metadata/checkpoint bytes without loading models or prediction values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    stage = Path(paths["stage70_worktree_read_only"])
    campaign = stage / "artifacts/stage70_76_architecture"
    role_path = campaign / "data_roles/DATA_ROLES.json"
    role = json.loads(role_path.read_text(encoding="utf-8"))
    claimed = role.pop("artifact_sha256")
    if object_digest(role) != claimed:
        raise ValueError("role content digest mismatch")
    roles = role["roles"]
    sets = [set(roles[name]) for name in ("original", "expansion", "confirmation", "protected")]
    if any(left & right for i, left in enumerate(sets) for right in sets[i + 1 :]):
        raise ValueError("owner roles overlap")
    checks = []
    for key in ("role_hash_resolution", "metadata_source", "exposure_ledger"):
        record = role[key]
        actual = sha256(Path(record["path"]))
        if actual != record["sha256"]:
            raise ValueError(f"role ancestor byte mismatch: {key}")
        checks.append({"dependency": key, **record, "hash_matches": True})
    charter_path = campaign / "time_charter/LABEL_TIME_CHARTER.json"
    charter = json.loads(charter_path.read_text(encoding="utf-8"))
    builder = charter["source_builder"]
    if sha256(Path(builder["path"])) != builder["sha256"]:
        raise ValueError("time charter release builder changed")
    ancestry_path = args.historical_root / "NESTED_ANCESTRY_AUDIT.json"
    ancestry = json.loads(ancestry_path.read_text(encoding="utf-8"))
    ancestry_hash = sha256(ancestry_path)
    table_index = json.loads(
        (args.historical_root / "FROZEN_EXPERT_TABLE_INDEX.json").read_text(encoding="utf-8")
    )
    if {row["ancestry_sha256"] for row in table_index} != {ancestry_hash}:
        raise ValueError("historical tables do not bind this ancestry audit")
    binding_checks = []
    for record in ancestry["input_bindings"].values():
        path = Path(record["path"])
        if path.suffix not in {".json", ".yaml", ".pt"}:
            continue
        if not path.is_file():
            binding_checks.append({**record, "status": "MISSING"})
            continue
        actual = sha256(path)
        binding_checks.append(
            {
                **record,
                "actual_sha256": actual,
                "status": "MATCH" if actual == record["sha256"] else "MISMATCH",
            }
        )
        if path.name in {"nested_protocol.json", "expert_artifact.json"}:
            content = json.loads(path.read_text(encoding="utf-8"))
            if not verify_artifact_hash(content):
                raise ValueError(f"producer content digest mismatch: {path}")
    producers = ancestry["producers"]
    expected = {
        (f, e, r)
        for f in range(3)
        for e in ("A5", "C2F", "PAIR")
        for r in ("inner0", "inner1", "inner2", "outer_dev")
    }
    if {(p["outer_fold"], p["expert"], p["role"]) for p in producers} != expected:
        raise ValueError("incomplete producer family inventory")
    for producer in producers:
        if producer["expert"] != "PAIR":
            split = producer["split_validation"]
            train = set(split["train_sequence_ids"])
            excluded = set(split["dev_sequence_ids"]) | set(
                split["excluded_outer_dev_sequence_ids"]
            )
            if train & excluded:
                raise ValueError("producer train/excluded overlap")
    output = {
        "artifact_type": "simplex_t_discovered_interfaces_v1",
        "role_manifest": {"path": str(role_path), "sha256": sha256(role_path)},
        "role_content_hash_verified": True,
        "role_dependencies": checks,
        "role_counts": {
            k: len(roles[k]) for k in ("original", "expansion", "confirmation", "protected")
        },
        "owner_authority_evidence": {
            "publisher": str(stage / "scripts/prepare_stage70_roles.py"),
            "publisher_refuses_regeneration": True,
            "role_resolution_pre_fit": role["optimizer_updates_at_assignment"] == 0,
            "adoption": "AWAITING_OWNER_SHARED_ACK",
        },
        "time_charter": {"path": str(charter_path), "sha256": sha256(charter_path)},
        "time_status": charter["status"],
        "time_builder_verified": builder,
        "time_consumer": str(stage / "scripts/build_stage71_counts.py"),
        "producer_ancestry": {"path": str(ancestry_path), "sha256": ancestry_hash},
        "producer_count": len(producers),
        "producer_inventory": producers,
        "physical_binding_checks": binding_checks,
        "all_checked_bindings_match": all(r["status"] == "MATCH" for r in binding_checks),
        "checked_checkpoint_count": sum(Path(r["path"]).suffix == ".pt" for r in binding_checks),
        "scope": "Metadata and checkpoint byte checks, not new replay or complete history proof",
        "prediction_values_read": False,
        "scientific_updates": 0,
        "new_architecture_scores_read": False,
    }
    write_new_json(args.output, output)
    print(
        json.dumps(
            {
                k: output[k]
                for k in (
                    "producer_count",
                    "checked_checkpoint_count",
                    "all_checked_bindings_match",
                    "role_counts",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
