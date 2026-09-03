#!/usr/bin/env python
"""Generate the closed JSON Schemas for E-Clock X0 runtime artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
NONEMPTY = {"type": "string", "minLength": 1}
ARMS = {"enum": ["X0-A5-REPLAY", "X0-PAIR-U", "X0-BASE-U", "X0-DYN-U"]}


def closed(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": required or list(properties),
        "properties": properties,
    }


def signed(properties: dict[str, Any]) -> dict[str, Any]:
    return closed({**properties, "artifact_sha256": SHA})


identity = closed(
    {
        "git_commit_observed": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "git_dirty_observed": {"const": False},
        "arm_id": {"enum": ["X0-PAIR-U", "X0-BASE-U", "X0-DYN-U"]},
        "scientific_role": NONEMPTY,
        "reference_family": {
            "type": ["string", "null"],
            "enum": ["official_a5_oof", None],
        },
        "seed": {"const": 7},
        "outer_fold": {"enum": [0, 1, 2]},
        "motion_feature_mode": {
            "enum": ["embedded_a5", "global_uniform_zeroed_control", "global_uniform"]
        },
        "model_class": NONEMPTY,
        "model_topology_sha256": SHA,
        "initialization_sha256": SHA,
        "config_path": NONEMPTY,
        "config_sha256": SHA,
        "protocol_path": NONEMPTY,
        "protocol_sha256": SHA,
        "reference_path": NONEMPTY,
        "reference_sha256": SHA,
        "split_manifest_path": NONEMPTY,
        "split_manifest_sha256": SHA,
        "cache_manifest_path": NONEMPTY,
        "cache_manifest_sha256": SHA,
        "ordered_token_identity_sha256": SHA,
        "target_sha256": SHA,
        "fold_assignment_sha256": SHA,
        "sample_weight_sha256": SHA,
        "train_token_subset_sha256": SHA,
        "dev_token_subset_sha256": SHA,
        "optimizer_config": closed(
            {
                "name": {"const": "AdamW"},
                "learning_rate": {"type": "number", "exclusiveMinimum": 0},
                "weight_decay": {"type": "number", "minimum": 0},
            }
        ),
        "scheduler_config": closed({"name": {"const": "constant"}}),
        "precision_mode": {"const": "float32"},
        "update_budget": {"type": "integer", "minimum": 1},
        "checkpoint_policy": {"const": "last_update_fixed_budget"},
    }
)

physical = closed(
    {
        "path": NONEMPTY,
        "file_sha256": SHA,
        "artifact_sha256": SHA,
        "bytes": {"type": "integer", "minimum": 1},
        "producer_identity": NONEMPTY,
    }
)

row = closed(
    {
        "sample_token": NONEMPTY,
        "sequence_id": NONEMPTY,
        "track_id": NONEMPTY,
        "outer_fold": {"enum": [0, 1, 2]},
        "target_ttc_s": {"type": "number"},
        "target_benchmark_phase": {"type": "number"},
        "predicted_benchmark_phase": {"type": "number"},
        "predicted_inverse_ttc_raw": {"type": "number"},
        "predicted_ttc_raw": {"type": "number"},
        "predicted_ttc_clipped": {"type": "number"},
        "is_clip_saturated": {"type": "boolean"},
        "scientific_mid_per_row": {"type": "number", "minimum": 0},
        "scientific_failure": {"type": "boolean"},
        "sample_weight": {"type": "number", "exclusiveMinimum": 0},
        "arm_id": ARMS,
        "seed": {"const": 7},
        "checkpoint_sha256": SHA,
        "config_sha256": SHA,
        "protocol_sha256": SHA,
        "cache_manifest_sha256": SHA,
        "split_manifest_sha256": SHA,
    }
)

schemas = {
    "run_manifest": signed(
        {
            "artifact_type": {"const": "eclock_x0_run_manifest_v2"},
            "arm_id": ARMS,
            "evidence_class": {"enum": ["dry_run", "scientific_oof"]},
            "scientific_result": {"type": "boolean"},
            "git_commit_observed": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
            "git_dirty_observed": {"type": "boolean"},
            "config_reference": physical,
            "protocol_reference": physical,
            "reference_reference": physical,
            "checkpoint_policy": {"const": "last_update_fixed_budget"},
        }
    ),
    "initialization_manifest": signed(
        {
            "artifact_type": {"const": "eclock_x0_initialization_manifest_v2"},
            "arm_id": ARMS,
            "seed": {"const": 7},
            "outer_fold": {"enum": [0, 1, 2]},
            "model_class": NONEMPTY,
            "model_topology_sha256": SHA,
            "initialization_sha256": SHA,
            "trainable_parameter_count": {"type": "integer", "minimum": 0},
        }
    ),
    "data_cache_binding": signed(
        {
            "artifact_type": {"const": "eclock_x0_data_cache_binding_v2"},
            "cache_manifest": physical,
            "preprocessing_version": NONEMPTY,
            "ordered_token_identity_sha256": SHA,
            "shards": {"type": "array", "minItems": 32, "maxItems": 32, "items": physical},
        }
    ),
    "split_binding": signed(
        {
            "artifact_type": {"const": "eclock_x0_split_binding_v2"},
            "split_manifest": physical,
            "fold_assignment_sha256": SHA,
            "train_token_subset_sha256": SHA,
            "dev_token_subset_sha256": SHA,
            "outer_fold": {"enum": [0, 1, 2]},
        }
    ),
    "checkpoint_manifest": signed(
        {
            "artifact_type": {"const": "eclock_x0_checkpoint_manifest_v2"},
            "checkpoint_policy": {"const": "last_update_fixed_budget"},
            "checkpoint_path": NONEMPTY,
            "checkpoint_file_sha256": SHA,
            "checkpoint_bytes": {"type": "integer", "minimum": 1},
            "completed_updates": {"type": "integer", "minimum": 1},
            "update_budget": {"type": "integer", "minimum": 1},
            "frozen": {"type": "boolean"},
            "scientific_identity": identity,
            "batch_schedule_sha256": SHA,
        }
    ),
    "resume_decision": signed(
        {
            "artifact_type": {"const": "eclock_x0_resume_decision_v2"},
            "decision": {"const": "resume_accepted"},
            "checkpoint_path": NONEMPTY,
            "checkpoint_physical_sha256": SHA,
            "checkpoint_manifest_sha256": SHA,
            "scientific_identity": identity,
            "completed_updates": {"type": "integer", "minimum": 1},
            "rng_state_sha256": closed(
                {"python": SHA, "numpy": SHA, "torch_cpu": SHA, "torch_cuda": SHA}
            ),
            "sampler_order_state": closed(
                {
                    "next_update": {"type": "integer", "minimum": 1},
                    "batch_count": {"type": "integer", "minimum": 1},
                }
            ),
        }
    ),
    "row_level_oof": row,
    "fold_summary": signed(
        {
            "artifact_type": {"const": "eclock_x0_fold_summary_v2"},
            "status": {"const": "completed_after_frozen_checkpoint"},
            "arm_id": ARMS,
            "outer_fold": {"enum": [0, 1, 2]},
            "seed": {"const": 7},
            "checkpoint_policy": {"const": "last_update_fixed_budget"},
            "checkpoint_path": NONEMPTY,
            "checkpoint_file_sha256": SHA,
            "checkpoint_manifest_sha256": SHA,
            "external_official_a5": {"type": "boolean"},
            "oof_path": NONEMPTY,
            "oof_file_sha256": SHA,
            "oof_bytes": {"type": "integer", "minimum": 1},
            "row_count": {"type": "integer", "minimum": 1},
            "outer_train_token_sha256": SHA,
            "outer_dev_token_sha256": SHA,
            "outer_dev_evaluations": {"const": 1},
            "outer_dev_used_during_training": {"const": False},
            "outer_dev_used_for_selection": {"const": False},
        }
    ),
    "bootstrap_artifact": signed(
        {
            "artifact_type": {"const": "eclock_x0_bootstrap_v2"},
            "method": {"const": "paired_hierarchical_sequence_then_track_cluster_bootstrap"},
            "cluster_order": {"const": ["sequence_id", "track_id"]},
            "rows_sampled_as_complete_tracks": {"const": True},
            "paired_identical_draws": {"const": True},
            "window_level_bootstrap_used": {"const": False},
            "seed": {"type": "integer"},
            "draws": {"type": "integer", "minimum": 1},
            "draws_identity_sha256": SHA,
            "token_count": {"type": "integer", "minimum": 1},
            "candidate_identity": closed(
                {
                    "reference_family": NONEMPTY,
                    "path": NONEMPTY,
                    "file_sha256": SHA,
                    "artifact_sha256": SHA,
                }
            ),
            "reference_identity": closed(
                {
                    "reference_family": {"const": "official_a5_oof"},
                    "path": NONEMPTY,
                    "file_sha256": SHA,
                    "artifact_sha256": SHA,
                }
            ),
            "candidate_mid": {"type": "object"},
            "reference_mid": {"type": "object"},
            "delta_candidate_minus_reference": {"type": "object"},
            "protocol_sha256": SHA,
        }
    ),
    "gate_decision": signed(
        {
            "artifact_type": {"const": "eclock_x0_gate_decision_v2"},
            "arm_id": ARMS,
            "reference_family": {"const": "official_a5_oof"},
            "reference_identity": physical,
            "decision": NONEMPTY,
            "checks": {"type": "object", "additionalProperties": {"type": "boolean"}},
        }
    ),
    "aggregate": signed(
        {
            "artifact_type": {"const": "eclock_x0_aggregate_v2"},
            "arm_id": ARMS,
            "evidence_class": {"const": "scientific_oof"},
            "scientific_result": {"const": True},
            "reference_family": {"const": "official_a5_oof"},
            "reference_identity": {"type": "object"},
            "config_sha256": SHA,
            "protocol_sha256": SHA,
            "reference_sha256": SHA,
            "checkpoint_sha256_by_fold": {"type": "object"},
            "metrics": {"type": "object"},
            "reference_metrics": {"type": "object"},
            "delta_mid_vs_official_a5_oof": {"type": "number"},
            "clipping_diagnostics": {"type": "object"},
            "bootstrap": {"type": "object"},
            "gate_decision": {"type": "object"},
            "integrity_chain_complete": {"const": True},
        }
    ),
    "dyn_w_not_executed": signed(
        {
            "artifact_type": {"const": "eclock_x0_dyn_w_not_executed_v2"},
            "arm_id": {"const": "X0-DYN-W"},
            "execution_authorized": {"const": False},
            "status": {"const": "not_executed"},
            "scientific_result": {"const": False},
            "loss_reduction": {"const": "normalized_weighted_absolute_phase_error"},
            "checkpoint_policy": {"const": "last_update_fixed_budget"},
            "forward_executed": {"const": False},
            "training_executed": {"const": False},
            "oof_executed": {"const": False},
        }
    ),
}


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "schemas"
    for name, schema in schemas.items():
        schema["$id"] = f"scientific_recovery_v9_eclock_{name}_v2.schema.json"
        path = root / schema["$id"]
        path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
