"""Build a zero-update feasibility plan for a complete, group-cross-fitted V16 refit.

This is an inventory and arithmetic planner, not a trainer or authorization receipt.
It never loads a checkpoint, sensor payload, test label or optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCERS = ("E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F")
PAIRS = ("PAIR_E_MATCHED", "PAIR_R")
CONTEXTS = ("E_CTX_MATCHED", "R_CTX")


def digest(path: Path) -> str:
    """Hash a source without retaining its contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read a metadata document."""
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    """Write plan evidence only to the requested output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def allocate(counts: dict[str, int]) -> list[list[str]]:
    """Balance three folds by query counts, never by labels or model errors."""
    folds: list[list[str]] = [[], [], []]
    totals = [0, 0, 0]
    ordered = sorted(
        counts,
        key=lambda g: (-counts[g], hashlib.sha256(f"v16_train40_seed7:{g}".encode()).hexdigest()),
    )
    for group in ordered:
        index = min(range(3), key=lambda i: (totals[i], len(folds[i]), i))
        folds[index].append(group)
        totals[index] += counts[group]
    return [sorted(groups) for groups in folds]


def chain(name: str, population: int, original_p: int, original_h: int) -> list[dict[str, Any]]:
    """Enumerate eight required trainable stages and population-scaled endpoints."""
    result = []
    for model in (*PRODUCERS, *PAIRS, *CONTEXTS):
        if model in PRODUCERS:
            updates, batch, parents = 18 * math.ceil(population / 32), 32, []
        elif model in PAIRS:
            updates, batch = math.ceil(6840 * population / original_p), 32
            parents = ["E_A5_MATCHED" if model.startswith("PAIR_E") else "R_A5"]
        else:
            updates, batch = math.ceil(2500 * population / original_h), 128
            parents = (
                list(PRODUCERS[:2]) + [PAIRS[0]]
                if model.startswith("E_")
                else (list(PRODUCERS[2:]) + [PAIRS[1]])
            )
        result.append(
            {
                "id": f"{name}/{model}",
                "train_population": population,
                "scientific_updates_proposed": updates,
                "batch_size": batch,
                "depends_on": [f"{name}/{parent}" for parent in parents],
                "fresh_initialization": True,
                "seed": 7,
            }
        )
    return result


def build(workspace: Path, output: Path) -> dict[str, Any]:
    """Check role identities and produce a proposed dependency and exposure plan."""
    run13 = workspace / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008"
    native = workspace / "e-jepa-ttc-v12-efficient-context/artifacts/train40_system_20261005"
    sources, roles, parts = {}, {}, []
    for role in ("P", "H", "V"):
        manifest_path = run13 / f"{role}_MANIFEST.json"
        manifest = read(manifest_path)
        rows_path = Path(manifest["rows_path"])
        if digest(rows_path) != manifest["rows_sha256"]:
            raise ValueError("Historical metadata changed")
        frame = pd.read_parquet(rows_path, columns=["sample_token", "group_id"])
        if frame.sample_token.duplicated().any():
            raise ValueError("Duplicate role queries")
        groups = sorted(frame.group_id.unique().tolist())
        if groups != sorted(manifest["group_ids"]) or len(frame) != manifest["population_size"]:
            raise ValueError("Role population differs")
        roles[role] = {"queries": len(frame), "groups": groups}
        parts.append(frame)
        sources[str(manifest_path)] = digest(manifest_path)
        sources[str(rows_path)] = digest(rows_path)
    allrows = pd.concat(parts, ignore_index=True)
    if allrows.sample_token.duplicated().any() or sum(len(v["groups"]) for v in roles.values()) != (
        allrows.group_id.nunique()
    ):
        raise ValueError("Historical role overlap")
    native_path = native / "TRAIN40_ROWS.parquet"
    declared = pd.read_parquet(native_path, columns=["sample_token", "sequence_id"])
    if declared.sample_token.duplicated().any() or set(declared.sample_token) != set(
        allrows.sample_token
    ):
        raise ValueError("Union differs from native TRAIN40 query IDs")
    joined = allrows.merge(declared, on="sample_token", validate="one_to_one")
    if not (joined.group_id == joined.sequence_id).all():
        raise ValueError("Acquisition group mapping differs")
    sources[str(native_path)] = digest(native_path)
    teacher_manifest_path = native / "TEACHER_MANIFEST.json"
    teacher_receipt_path = native / "TEACHER_COMPLETE_ADMISSION.json"
    teacher_freeze_path = native / "TEACHER_FREEZE.json"
    teacher = read(teacher_manifest_path)
    teacher_receipt = read(teacher_receipt_path)
    if (
        teacher["rows_sha256"] != sources[str(native_path)]
        or teacher["row_count"] != len(allrows)
        or teacher_receipt["complete_rows"] != len(allrows)
        or teacher_receipt["status"] != "PASSED"
        or teacher["freeze_sha256"] != digest(teacher_freeze_path)
        or teacher_receipt["teacher_freeze_sha256"] != digest(teacher_freeze_path)
    ):
        raise ValueError("Historical teacher admission does not bind the TRAIN40 population")
    for path in (teacher_manifest_path, teacher_receipt_path, teacher_freeze_path):
        sources[str(path)] = digest(path)
    sources[str(ROOT / "operational/rgb_port/recipe.py")] = digest(
        ROOT / "operational/rgb_port/recipe.py"
    )
    sources[str(ROOT / "operational/rgb_port/train_heads.py")] = digest(
        ROOT / "operational/rgb_port/train_heads.py"
    )
    sources[str(Path(__file__).resolve())] = digest(Path(__file__))
    counts = {str(k): int(v) for k, v in allrows.groupby("group_id").size().items()}
    folds = allocate(counts)
    if set().union(*map(set, folds)) != set(counts) or sum(map(len, folds)) != len(counts):
        raise ValueError("Invalid fold coverage")
    tasks, fold_records = [], []
    for index, heldout in enumerate(folds):
        train = sorted(set(counts) - set(heldout))
        n = sum(counts[g] for g in train)
        fold_records.append(
            {
                "fold": index,
                "train_groups": train,
                "heldout_groups": heldout,
                "train_queries": n,
                "heldout_queries": len(allrows) - n,
            }
        )
        tasks += chain(f"fold{index}", n, roles["P"]["queries"], roles["H"]["queries"])
    final = chain("final", len(allrows), roles["P"]["queries"], roles["H"]["queries"])
    tasks += final
    head_updates = math.ceil(2500 * len(allrows) / roles["H"]["queries"])
    tasks.append(
        {
            "id": "PHASE_ONLY_EVENT_S7_TRAIN40",
            "train_population": len(allrows),
            "scientific_updates_proposed": head_updates,
            "batch_size": 128,
            "depends_on": [f"fold{i}/{m}" for i in range(3) for m in CONTEXTS],
            "fresh_initialization": True,
            "seed": 7,
            "input": "sealed complete OOF states",
        }
    )
    total = sum(task["scientific_updates_proposed"] for task in tasks)
    value = {
        "status": "PROPOSED_COMPLETE_REFIT_NOT_EXECUTABLE",
        "training_started": False,
        "optimizer_updates_this_task": 0,
        "execution_enabled": False,
        "submission_enabled": False,
        "family": "V16 PHASE_ONLY RGB+event, seed7, no ensemble",
        "population": {"queries": len(allrows), "groups": len(counts), "roles": roles},
        "sources": sources,
        "teacher_cache": {
            "status": "HISTORICAL_COMPLETE_RECEIPT_BOUND_NOT_REHASHED",
            "rows": teacher["row_count"],
            "shards": teacher_receipt["shards_verified"],
            "bytes_previously_verified": teacher_receipt["verified_bytes"],
            "generic_weights_sha256": teacher_receipt["ordered_teacher_weights_sha256"],
            "new_inference": False,
            "next_check": "Verify current shard integrity and fold token mapping before reuse",
        },
        "folds": fold_records,
        "trainable_tasks": tasks,
        "stacking": {
            "strategy": "Three complete lower-chain group folds plus all-TRAIN final chain",
            "fold_chain": "Fresh producers -> PAIR -> fold normalizers -> CTX -> heldout states",
            "within_fold_upper_training": "Sequential in-sample; not OOF within train groups",
            "OOF_boundary": "Every learned ancestor excludes the output query acquisition group",
            "historical_supervised_parents_allowed_for_folds": False,
            "head": "Fresh PHASE_ONLY on heldout-fold states; recompute group-balanced mass",
            "deployment": "One all-TRAIN chain plus OOF-trained head; no fold ensemble",
            "risk": "OOF-to-final features can shift; improvement is not guaranteed",
            "evaluation": "OOF states are head TRAIN; fitted head scores on them are not test",
            "teachers": "Generic frozen DINO on train pixels; verify coverage and ancestry",
        },
        "exposure": {
            "producers": "18 epochs, effective batch32, fresh initialization; local recipe",
            "PAIR": "ceil(6840*N/53890), batch32; grouped sampler; approximate exposure",
            "CTX_and_head": "ceil(2500*N/18422), batch128; recompute group mass on fit population",
            "schedulers": "Producer epochs; constant PAIR LR; scale CTX/head warmup and cosine",
            "checkpoint": "Fixed final endpoint, no selecting checkpoints on H/V/EvTTC/test12",
        },
        "budget": {
            "fit_count": len(tasks),
            "scientific_updates_proposed": total,
            "producer_updates": sum(
                t["scientific_updates_proposed"]
                for t in tasks
                if t["id"].split("/")[-1] in PRODUCERS
            ),
            "single_chain_in_sample_alternative_updates": sum(
                t["scientific_updates_proposed"] for t in final
            )
            + head_updates,
            "single_chain_alternative_fit_count": 9,
            "alternative_selected": False,
            "alternative_limit": "Cheaper declared in-sample stacking; no OOF head inputs",
            "GPU_hours": None,
            "technical_and_recovery_cap": None,
            "total_storage_bytes": None,
            "status": "Update arithmetic; no measured resource budget or execution authorization",
            "profiling_needed": "Fixtures per modality/producer type; extraction and storage",
        },
        "excluded_fits": ["E_H1_MATCHED", "R_H1", "F_TRUE", "F_ZERO", "C1", "A1", "V19"],
        "readiness": {
            "P_H_V_union_and_group_identity": "PASS",
            "raw_media_full_validation": "NOT_REPEATED",
            "teacher_all_TRAIN40_coverage": "HISTORICAL_COMPLETE_RECEIPT_BOUND_NOT_REHASHED",
            "temporal_history_eligibility": "UNCONFIRMED",
            "full_refit_training_adapter": "NOT_IMPLEMENTED",
            "CPU_GPU_resume_and_export_validation": "NOT_EXECUTED_FOR_NEW_CHAIN",
            "historical_trainer_compatibility": (
                "Old trainers enforce P/H and fixed endpoints; cannot launch this unchanged"
            ),
        },
        "delivery_order": [
            "Confirm RGB+event causal history eligibility using the draft question",
            "Bind raw input and teacher coverage, licenses, generic teacher and source hashes",
            "Implement separate refit adapter and role contracts; preserve historical runners",
            "Measure technical runtime/storage, set complete cap and exact resumption tests",
            "Register selected strategy and budget before optimizer updates",
            "Fit complete folds, seal OOF states; fit final chain and PHASE_ONLY head",
            "Freeze endpoint, verify full sensor/export parity and coverage",
            "Generate one final test12 package; manual user submission per existing instruction",
        ],
        "official_history_question": (
            "Does the RGB+event track allow causal same-object history across samples, up to "
            "eight observations and 650 ms, without future sensors or test TTC? "
            "Does it permit generic frozen DINO representation supervision during TRAIN only? "
            "How should these be declared in the factsheet?"
        ),
    }
    write(output / "FULL_REFIT_PLAN.json", value)
    write(output / "TRAIN40_GROUPS_AND_FOLDS.json", {"query_counts": counts, "folds": fold_records})
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT.parent)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/v16_train40_refit_20261017")
    args = parser.parse_args()
    result = build(args.workspace.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "queries": result["population"]["queries"],
                "groups": result["population"]["groups"],
                "budget": result["budget"],
            }
        )
    )
