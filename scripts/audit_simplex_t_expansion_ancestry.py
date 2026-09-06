"""Recheck frozen inner producer ancestry for D1 without fitting or reading new scores."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.data.nested_provenance import (
    validate_frozen_teacher_dependency,
    validate_uninitialized_producer_contract,
)
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.training.incremental_residual import deterministic_sequence_grouped_schedule


def main() -> None:
    """Verify all nine inner families against expansion and outer-held-out groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    root = Path(paths["worktree"])
    config = json.loads(
        (root / "configs/experiment/simplex_t_coordination.json").read_text(encoding="utf-8")
    )
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    reference = ack["producers"]["authoritative_historical_manifest"]
    ancestry = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
    bindings = {
        Path(value["path"]).resolve(): value for value in ancestry["input_bindings"].values()
    }
    checked = {}

    def verify(path: Path) -> Path:
        path = path.resolve(strict=True)
        expected = bindings[path]["sha256"]
        if compute_file_hash(str(path)) != expected:
            raise ValueError(f"frozen ancestor bytes changed: {path.name}")
        checked[str(path)] = expected
        return path

    teacher_path = next(path for path in bindings if path.name == "FROZEN_TEACHER_PROVENANCE.json")
    teacher = json.loads(verify(teacher_path).read_text(encoding="utf-8"))
    verify(Path(teacher["teacher_manifest_path"]))
    roles = ack["interfaces"]["role_manifest"]["roles"]
    expansion = set(roles["expansion"])
    original = set(roles["original"])
    producers = {(p["outer_fold"], p["role"], p["expert"]): p for p in ancestry["producers"]}
    checkpoint_paths = {
        value["sha256"]: path for path, value in bindings.items() if path.suffix == ".pt"
    }
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    records = []
    for outer in range(3):
        for inner in range(3):
            family = f"inner{inner}"
            train_sets = []
            for expert in ("A5", "C2F"):
                if not admitted([root])["has_headroom"]:
                    raise RuntimeError("RESOURCE_PAUSE before CPU ancestry inspection")
                producer = producers[outer, family, expert]
                checkpoint = verify(checkpoint_paths[producer["checkpoint_sha256"]])
                directory = checkpoint.parent.parent
                protocol = json.loads(
                    verify(directory / "nested_protocol.json").read_text(encoding="utf-8")
                )
                effective = yaml.safe_load(
                    verify(directory / "effective_config.yaml").read_text(encoding="utf-8")
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                initialization = validate_uninitialized_producer_contract(
                    payload, effective, protocol
                )
                validate_frozen_teacher_dependency(initialization, teacher)
                del payload
                split = protocol["folds"][0]
                train = set(split["train_sequence_ids"])
                outer_dev = set(producer["split_validation"]["excluded_outer_dev_sequence_ids"])
                if (
                    not train <= original
                    or train & (expansion | outer_dev)
                    or train != set(producer["split_validation"]["train_sequence_ids"])
                ):
                    raise ValueError("expansion or outer-dev appears in fitted producer ancestry")
                train_sets.append(train)
            if train_sets[0] != train_sets[1]:
                raise ValueError("A5 and C2F family training sets disagree")
            pair = producers[outer, family, "PAIR"]
            checkpoint = verify(checkpoint_paths[pair["checkpoint_sha256"]])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            identity = payload["identity"]
            if (
                identity["outer_fold"] != outer
                or identity["inner_fold"] != inner
                or identity["role"] != "inner_oof"
            ):
                raise ValueError("PAIR identity mismatch")
            feature_dir = checkpoint.parents[5] / "feature_cache"
            feature = json.loads(
                verify(feature_dir / f"outer{outer}_inner{inner}.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                not verify_artifact_hash(feature)
                or feature["artifact_sha256"] != identity["feature_cache_artifact_sha256"]
                or feature["identity"]["a5_checkpoint_sha256"]
                != producers[outer, family, "A5"]["checkpoint_sha256"]
                or pair["nested_a5_ancestor_sha256"] != feature["identity"]["a5_checkpoint_sha256"]
            ):
                raise ValueError("PAIR nested teacher feature lineage mismatch")
            metadata = pd.read_csv(
                verify(feature_dir / feature["metadata"]["path"]),
                usecols=["sample_token", "sequence_id"],
            )
            if set(metadata.sequence_id) & expansion or not set(metadata.sequence_id) <= original:
                raise ValueError("PAIR source universe contains expansion or closed groups")
            train_meta = metadata.loc[metadata.sequence_id.isin(train_sets[0])]
            recipe = identity["config"]
            _, schedule = deterministic_sequence_grouped_schedule(
                train_meta.sequence_id.astype(str).tolist(),
                train_meta.sample_token.astype(str).tolist(),
                seed=recipe["seed"],
                batch_size=recipe["batch_size"],
                updates=recipe["update_budget"],
            )
            if (
                schedule != identity["batch_schedule_sha256"]
                or schedule != pair["train_token_schedule_sha256"]
                or payload["completed_updates"] != recipe["update_budget"]
            ):
                raise ValueError("PAIR fitted schedule or endpoint differs from verified ancestry")
            records.append(
                {
                    "outer": outer,
                    "inner": inner,
                    "fitted_sequences": sorted(train_sets[0]),
                    "excluded_expansion_sequences": sorted(expansion),
                    "pair_train_schedule_sha256": schedule,
                    "checkpoints": {
                        e: producers[outer, family, e]["checkpoint_sha256"]
                        for e in ("A5", "C2F", "PAIR")
                    },
                }
            )
            del payload
            gc.collect()
    write_new_json(
        args.output,
        {
            "status": "D1_FROZEN_INNER_ANCESTRY_RECHECKED_NO_EXPERT_REFITS",
            "families": records,
            "verified_bindings": checked,
            "authoritative_ancestry": reference,
            "new_optimizer_updates": 0,
            "stage70_scores_read": False,
            "teacher_scope": teacher["scope"],
            "external_teacher_weights_rehashed": False,
            "teacher_weights_not_needed_by_frozen_expert_inference": True,
            "claim_limit": "Sequence-level exclusions; acquisition identity remains unverified",
            "not_yet_extraction_authorization": True,
        },
    )
    print(json.dumps({"families_verified": len(records), "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
