"""Count a prospective nested grouped refit without training or reading TTC targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def digest(path: Path) -> str:
    """Hash a small metadata input."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_plan(workspace: Path) -> dict[str, Any]:
    """Build five group blocks, lineage exclusions and exact update counts."""
    source = workspace / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008"
    roles: dict[str, dict[str, int]] = {}
    bindings: dict[str, str] = {}
    for role in ("P", "H", "V"):
        manifest_path = source / f"{role}_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows_path = Path(manifest["rows_path"])
        if digest(rows_path) != manifest["rows_sha256"]:
            raise ValueError(f"Changed {role} metadata")
        rows = pd.read_parquet(rows_path, columns=["sample_token", "group_id"])
        if rows.sample_token.duplicated().any():
            raise ValueError("Duplicate query identity")
        roles[role] = {str(k): int(v) for k, v in rows.groupby("group_id").size().items()}
        bindings[str(manifest_path)] = digest(manifest_path)
        bindings[str(rows_path)] = digest(rows_path)
    if tuple(len(roles[r]) for r in ("P", "H", "V")) != (24, 8, 8):
        raise ValueError("Historical P/H/V membership changed")
    all_groups = set().union(*(set(groups) for groups in roles.values()))
    if len(all_groups) != 40:
        raise ValueError("Historical groups overlap")
    counts = {group: n for groups in roles.values() for group, n in groups.items()}
    if sum(counts.values()) != 88_744:
        raise ValueError("TRAIN40 population changed")
    # Balance query counts with eight groups per block; never inspect targets/errors.
    ordered_p = sorted(
        roles["P"],
        key=lambda group: (
            -counts[group],
            hashlib.sha256(f"FINAL_OOF5_S7|{group}".encode()).hexdigest(),
        ),
    )
    p_blocks: list[list[str]] = [[], [], []]
    totals = [0, 0, 0]
    for group in ordered_p:
        index = min((i for i in range(3) if len(p_blocks[i]) < 8), key=lambda i: (totals[i], i))
        p_blocks[index].append(group)
        totals[index] += counts[group]
    blocks = [sorted(roles["V"]), sorted(roles["H"]), *map(sorted, p_blocks)]
    rotations = []
    for number in range(5):
        generation = blocks[number]
        refiner = blocks[(number + 1) % 5]
        producer = sorted(all_groups - set(generation) - set(refiner))
        if set(producer) & set(refiner) or set(producer + refiner) & set(generation):
            raise ValueError("A generation block entered learned ancestry")
        np_, nh, nv = (sum(counts[g] for g in names) for names in (producer, refiner, generation))
        updates = {
            **{name: 18 * math.ceil(np_ / 32) for name in ("E_A5", "E_C2F", "R_A5", "R_C2F")},
            **{name: math.ceil(6_840 * np_ / 53_890) for name in ("PAIR_E", "PAIR_R")},
            **{name: math.ceil(2_500 * nh / 18_422) for name in ("E_CTX", "R_CTX")},
        }
        rotations.append(
            {
                "rotation": number,
                "producer_groups": producer,
                "refiner_groups": refiner,
                "generation_groups": generation,
                "queries": {"producer_pair": np_, "refiner": nh, "generation": nv},
                "updates": updates,
                "total_updates": sum(updates.values()),
                "reuse": "CANDIDATE_V13_P_H_V_REQUIRES_FULL_BINDING_AUDIT" if number == 0 else None,
                "generation_ancestry_excludes_group": True,
            }
        )
    final_updates = {
        **{name: 49_932 for name in ("E_A5", "E_C2F", "R_A5", "R_C2F")},
        **{name: 11_264 for name in ("PAIR_E", "PAIR_R")},
        **{name: 12_044 for name in ("E_CTX", "R_CTX", "PHASE_ONLY")},
    }
    conceptual = sum(r["total_updates"] for r in rotations) + sum(final_updates.values())
    reused_updates = rotations[0]["total_updates"] + 2 * 49_932
    return {
        "schema": "final_refit_nested_rotation_review_v1",
        "status": "PROSPECTIVE_COST_AND_ANCESTRY_COMPARISON_NOT_AUTHORIZED_QUEUE",
        "execution_enabled": False,
        "scientific_updates_executed": 0,
        "partition": (
            "P descending query count, ties SHA256 FINAL_OOF5_S7; assign to lowest-count "
            "block with capacity eight, ties block index; no TTC labels or scores"
        ),
        "group_blocks": {f"B{i}": groups for i, groups in enumerate(blocks)},
        "query_counts": counts,
        "metadata_bindings": bindings,
        "rotations": rotations,
        "final_updates": final_updates,
        "conceptual_fits": 49,
        "conceptual_updates": conceptual,
        "conditional_reused_fits": 10,
        "conditional_reused_updates": reused_updates,
        "conditional_new_fits": 39,
        "conditional_new_updates": conceptual - reused_updates,
        "sequential_alternative_new_fits": 7,
        "sequential_alternative_new_updates": 158_524,
        "pending_before_execution": [
            "Confirm admissibility of actual 650ms input/ROI/modality contract",
            "Admit exact rotation0 ancestry, sources, normalizers and generation identities",
            "Complete group-filtered producer and refiner runners with resume proof",
            "Budget feature extraction, training, storage and recovery with scoped measurements",
            "Approve exact new resource budget",
        ],
        "construction": {
            "final_context_inputs": (
                "Stitched raw producer/PAIR OOF PHASE17, then one OOF TRAIN40 normalizer"
            ),
            "rotation_normalizers": "Fit only the corresponding eight refiner-training groups",
            "final_phase_inputs": (
                "Stitched context OOF outputs excluding each generated group's full ancestry"
            ),
            "final_pair_inputs": (
                "Native final A5 outputs over TRAIN40, sequential lower-stage block"
            ),
            "pair_reason": (
                "Do not mix token128 latent coordinates from independently fitted encoders; "
                "the OOF boundary is the physically defined PHASE17 interface"
            ),
            "deployment": "One final chain, no rotation ensemble",
            "limits": "OOF-to-final distribution shift remains; no guaranteed accuracy gain",
        },
    }


def main() -> None:
    """Write one prospective machine-readable comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: plan[k] for k in ("conditional_new_fits", "conditional_new_updates")}))


if __name__ == "__main__":
    main()
