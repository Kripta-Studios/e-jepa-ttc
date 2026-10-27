"""Materialize a label-blind grouped OOF contract for the V16 refit.

This module only creates row-reference manifests.  It never copies sensor
payloads, reads TTC targets, loads model weights, or performs optimizer work.
Each row keeps the historical role/local ordinal needed to address an existing
cache and the native ``source_ordinal`` needed to recover TRAIN40 order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROLES = ("P", "H", "V")
_ROTATION_ROLES = ("producer", "refiner", "generation")
_V13_RUN = Path("e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008")
_REUSE_FITS = {
    "E_A5_MATCHED": (30_330, "P"),
    "E_C2F_MATCHED": (30_330, "P"),
    "R_A5": (30_330, "P"),
    "R_C2F": (30_330, "P"),
    "PAIR_E_MATCHED": (6_840, "P"),
    "PAIR_R": (6_840, "P"),
    "E_CTX_MATCHED": (2_500, "H"),
    "R_CTX": (2_500, "H"),
}
_ROTATION0_AUDIT_SCHEMA = "final_refit_rotation0_reuse_audit_v2"


def sha256(path: Path) -> str:
    """Return the SHA-256 of a required file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_historical_rows(run: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Load only identity/lookup columns from the three historical roles."""
    frames: list[pd.DataFrame] = []
    manifests: dict[str, dict[str, Any]] = {}
    for historical_role in _ROLES:
        manifest_path = run / f"{historical_role}_MANIFEST.json"
        manifest = _json(manifest_path)
        rows_path = Path(str(manifest["rows_path"]))
        if sha256(manifest_path) == "" or sha256(rows_path) != manifest["rows_sha256"]:
            raise ValueError(f"Changed {historical_role} role metadata")
        if manifest.get("status") != "COMPLETE" or manifest.get("role") != historical_role:
            raise ValueError(f"Invalid {historical_role} role manifest")
        frame = pd.read_parquet(
            rows_path,
            columns=["ordinal", "source_ordinal", "sample_token", "group_id", "role"],
        ).rename(columns={"ordinal": "historical_ordinal"})
        if len(frame) != int(manifest["population_size"]):
            raise ValueError(f"{historical_role} population changed")
        if not (frame["role"].astype(str) == historical_role).all():
            raise ValueError(f"{historical_role} rows contain a different role")
        frame = frame.drop(columns=["role"])
        frame = frame.assign(
            historical_local_ordinal=pd.Series(range(len(frame)), dtype="int64"),
            historical_role=historical_role,
        )
        frames.append(frame)
        manifests[historical_role] = {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "rows_path": str(rows_path.resolve()),
            "rows_sha256": sha256(rows_path),
            "queries": len(frame),
            "groups": int(frame["group_id"].nunique()),
        }
    rows = pd.concat(frames, ignore_index=True)
    if rows["sample_token"].duplicated().any():
        raise ValueError("Historical roles overlap by sample token")
    if rows["source_ordinal"].duplicated().any():
        raise ValueError("Historical roles overlap by native source ordinal")
    if len(rows) != 88_744 or rows["group_id"].nunique() != 40:
        raise ValueError("Historical role union is not TRAIN40")
    return rows, manifests


def select_rotation_rows(rows: pd.DataFrame, rotation: dict[str, Any], role: str) -> pd.DataFrame:
    """Select one rotation role while preserving cache and native lookups."""
    if role not in _ROTATION_ROLES:
        raise ValueError(f"Unknown rotation role: {role}")
    groups = {str(value) for value in rotation[f"{role}_groups"]}
    selected = rows.loc[rows["group_id"].astype(str).isin(sorted(groups))].copy()
    selected.insert(0, "rotation", int(rotation["rotation"]))
    selected.insert(1, "rotation_role", role)
    return selected.sort_values("source_ordinal", kind="stable").reset_index(drop=True)


def validate_rotation_contract(rows: pd.DataFrame, review: dict[str, Any]) -> None:
    """Fail closed if blocks, counts, coverage, or label-blind policy changed."""
    if rows["sample_token"].duplicated().any():
        raise ValueError("Admitted rows contain duplicate sample tokens")
    if rows["source_ordinal"].duplicated().any():
        raise ValueError("Admitted rows contain duplicate native ordinals")
    if review.get("schema") != "final_refit_nested_rotation_review_v1":
        raise ValueError("Unexpected nested-rotation schema")
    policy = str(review.get("partition", ""))
    if "no TTC labels or scores" not in policy:
        raise ValueError("Partition is not explicitly label blind")
    blocks = review.get("group_blocks")
    rotations = review.get("rotations")
    if not isinstance(blocks, dict) or set(blocks) != {f"B{i}" for i in range(5)}:
        raise ValueError("Expected exactly five named blocks")
    block_sets = [{str(group) for group in blocks[f"B{i}"]} for i in range(5)]
    if any(len(groups) != 8 for groups in block_sets):
        raise ValueError("Every OOF block must contain exactly eight groups")
    if len(set().union(*block_sets)) != 40 or sum(map(len, block_sets)) != 40:
        raise ValueError("OOF blocks overlap or omit a TRAIN40 group")
    if not isinstance(rotations, list) or len(rotations) != 5:
        raise ValueError("Expected exactly five rotations")
    all_groups = {str(group) for group in rows["group_id"].unique()}
    if set().union(*block_sets) != all_groups:
        raise ValueError("Review groups differ from admitted role rows")
    token_union: set[str] = set()
    for number, rotation in enumerate(rotations):
        if int(rotation.get("rotation", -1)) != number:
            raise ValueError("Rotation numbering changed")
        producer = {str(value) for value in rotation["producer_groups"]}
        refiner = {str(value) for value in rotation["refiner_groups"]}
        generation = {str(value) for value in rotation["generation_groups"]}
        if (len(producer), len(refiner), len(generation)) != (24, 8, 8):
            raise ValueError("A rotation is not 24/8/8")
        if producer & refiner or producer & generation or refiner & generation:
            raise ValueError("Rotation roles overlap by group")
        if producer | refiner | generation != all_groups:
            raise ValueError("Rotation roles do not cover TRAIN40")
        for role, groups in zip(_ROTATION_ROLES, (producer, refiner, generation), strict=True):
            selected = rows.loc[rows["group_id"].astype(str).isin(sorted(groups))]
            expected_key = "producer_pair" if role == "producer" else role
            if len(selected) != int(rotation["queries"][expected_key]):
                raise ValueError(f"Rotation {number} {role} query count changed")
        generated = rows.loc[rows["group_id"].astype(str).isin(sorted(generation)), "sample_token"]
        current_tokens = set(generated.astype(str))
        if token_union & current_tokens:
            raise ValueError("A token is generated by more than one rotation")
        token_union |= current_tokens
    if token_union != set(rows["sample_token"].astype(str)):
        raise ValueError("Generation folds do not cover every token exactly once")


def _audit_endpoint(fit_dir: Path, expected_updates: int, expected_role: str) -> dict[str, Any]:
    receipt_path = fit_dir / "CHECKPOINT_RECEIPT.json"
    provenance_path = fit_dir / "RUN_PROVENANCE.json"
    journal_path = fit_dir / "UPDATE_JOURNAL.json"
    freeze_path = fit_dir / "ENDPOINT_FREEZE.json"
    checkpoint_path = fit_dir / "checkpoint_last.pt"
    receipt = _json(receipt_path)
    provenance = _json(provenance_path)
    journal = _json(journal_path)
    freeze = _json(freeze_path)
    fit_id = fit_dir.name
    if freeze.get("fit_id") != fit_id:
        raise ValueError(f"{fit_id} endpoint freeze identity mismatch")
    for name, path in {
        "CHECKPOINT_RECEIPT.json": receipt_path,
        "RUN_PROVENANCE.json": provenance_path,
        "UPDATE_JOURNAL.json": journal_path,
        "checkpoint_last.pt": checkpoint_path,
    }.items():
        if freeze.get("files", {}).get(name) != sha256(path):
            raise ValueError(f"{fit_id} changed frozen file: {name}")
    checkpoint_hash = sha256(checkpoint_path)
    completed = int(receipt.get("completed_updates", -1))
    if (
        receipt.get("status") != "COMPLETE"
        or not receipt.get("complete_state")
        or not receipt.get("scientific_endpoint")
        or completed != expected_updates
        or int(provenance.get("completed_updates", completed)) != expected_updates
        or int(journal.get("completed_updates", -1)) != expected_updates
        or int(journal.get("durable_updates", -1)) != expected_updates
        or receipt.get("checkpoint_sha256") != checkpoint_hash
        or provenance.get("checkpoint_sha256", checkpoint_hash) != checkpoint_hash
        or provenance.get("status") != "COMPLETE"
        or int(provenance.get("seed", -1)) != 7
    ):
        raise ValueError(f"{fit_id} is not the expected complete seed-7 endpoint")
    source_identity = provenance.get("source_identity", {})
    observed_role = source_identity.get("role")
    if observed_role is None:
        observed_role = "P" if expected_role == "P" else "H"
    if observed_role != expected_role:
        raise ValueError(f"{fit_id} does not bind historical role {expected_role}")
    return {
        "fit_id": fit_id,
        "updates": expected_updates,
        "historical_role": expected_role,
        "checkpoint_sha256": checkpoint_hash,
        "receipt_sha256": sha256(receipt_path),
        "provenance_sha256": sha256(provenance_path),
        "endpoint_freeze_sha256": sha256(freeze_path),
    }


def _audit_context_closure(
    run: Path,
    *,
    fit_id: str,
    prefix: str,
    modality: str,
    endpoints: dict[str, dict[str, Any]],
    h_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Prove the complete producer/PAIR -> observation -> normalizer -> H chain."""
    feature_dir = run / "features" / "H"
    history_receipt_path = feature_dir / f"{prefix}_HISTORY.json"
    history_receipt = _json(history_receipt_path)
    history_path = Path(str(history_receipt["output"]))
    observation_receipt_path = feature_dir / f"{prefix}_OBSERVATIONS.json"
    observation_receipt = _json(observation_receipt_path)
    observation_path = Path(str(observation_receipt["output"]))
    normalizer_path = feature_dir / f"{prefix}_NORMALIZER.json"
    normalizer = _json(normalizer_path)
    provenance = _json(run / "fits" / fit_id / "RUN_PROVENANCE.json")
    freeze_path = run / "rounds" / "PRODUCERS_PAIR_ENDPOINT_FREEZE.json"
    freeze = _json(freeze_path)

    expected_endpoint_hashes = {
        name: endpoint["checkpoint_sha256"]
        for name, endpoint in endpoints.items()
        if endpoint["historical_role"] == "P"
    }
    frozen_endpoints = freeze.get("endpoints")
    if freeze.get("schema") != "rgb_port_parent_round_freeze_v1" or not isinstance(
        frozen_endpoints, dict
    ):
        raise ValueError("rotation-0 producer/PAIR round freeze is invalid")
    observed_frozen_hashes = {
        str(name): value.get("checkpoint_sha256")
        for name, value in frozen_endpoints.items()
        if isinstance(value, dict)
    }
    if observed_frozen_hashes != expected_endpoint_hashes:
        raise ValueError("rotation-0 round freeze does not close over every P endpoint")
    freeze_sha = sha256(freeze_path)

    if (
        observation_receipt.get("schema") != "rgb_port_observation_features_v1"
        or observation_receipt.get("status") != "COMPLETE"
        or observation_receipt.get("role") != "H"
        or observation_receipt.get("modality") != modality
        or observation_receipt.get("endpoint_hashes") != expected_endpoint_hashes
        or observation_receipt.get("output_sha256") != sha256(observation_path)
    ):
        raise ValueError(f"{fit_id} observation receipt does not prove endpoint closure")

    with np.load(observation_path, allow_pickle=False) as archive:
        required = {
            "observation_id",
            "features",
            "anchor_us",
            "available_us",
            "role",
            "schema_sha256",
            "endpoint_freeze_sha256",
        }
        if set(archive.files) != required:
            raise ValueError(f"{fit_id} observation payload schema differs")
        observation_ids = np.asarray(archive["observation_id"]).astype(str)
        features = np.asarray(archive["features"], dtype=np.float32)
        anchors = np.asarray(archive["anchor_us"], dtype=np.int64)
        available = np.asarray(archive["available_us"], dtype=np.int64)
        roles = np.asarray(archive["role"]).astype(str)
        feature_schema = str(archive["schema_sha256"])
        producer_sha = str(archive["endpoint_freeze_sha256"])
    observation_rows = int(observation_receipt.get("rows", -1))
    if (
        len(observation_ids) != observation_rows
        or len(set(observation_ids.tolist())) != observation_rows
        or features.shape != (observation_rows, 17)
        or not np.isfinite(features).all()
        or anchors.shape != (observation_rows,)
        or available.shape != (observation_rows,)
        or np.any(available < anchors)
        or set(roles.tolist()) != {"H"}
        or producer_sha != freeze_sha
    ):
        raise ValueError(f"{fit_id} observation payload identity or clocks differ")

    unique_ids_sha = hashlib.sha256(
        "\n".join(sorted(observation_ids.tolist())).encode()
    ).hexdigest()
    mean = np.asarray(normalizer.get("mean", []), dtype=np.float64)
    scale = np.asarray(normalizer.get("scale", []), dtype=np.float64)
    if (
        normalizer.get("schema") != "rgb_port_normalizer_v1"
        or normalizer.get("fit_role") != "H"
        or normalizer.get("modality") != modality
        or normalizer.get("schema_sha256") != feature_schema
        or normalizer.get("producer_sha256") != producer_sha
        or normalizer.get("unique_ids_sha256") != unique_ids_sha
        or mean.shape != (17,)
        or scale.shape != (17,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError(f"{fit_id} normalizer does not prove H-only observation identity")

    if (
        history_receipt.get("schema") != "rgb_port_history_features_v1"
        or history_receipt.get("status") != "COMPLETE"
        or history_receipt.get("role") != "H"
        or history_receipt.get("modality") != modality
        or int(history_receipt.get("queries", -1)) != len(h_rows)
        or history_receipt.get("source_sha256") != observation_receipt["output_sha256"]
        or history_receipt.get("normalizer_sha256") != sha256(normalizer_path)
        or history_receipt.get("output_sha256") != sha256(history_path)
        or provenance.get("source_sha256") != history_receipt["output_sha256"]
        or provenance.get("normalizer_sha256") != sha256(normalizer_path)
    ):
        raise ValueError(f"{fit_id} history/normalizer source is inconsistent")

    with np.load(history_path, allow_pickle=False) as archive:
        required = {"sample_token", "group_id", "role"}
        if not required.issubset(archive.files):
            raise ValueError(f"{fit_id} history lacks H identity fields")
        history_tokens = np.asarray(archive["sample_token"]).astype(str)
        history_groups = np.asarray(archive["group_id"]).astype(str)
        history_roles = np.asarray(archive["role"]).astype(str)
    expected_tokens = h_rows["sample_token"].astype(str).tolist()
    expected_groups = h_rows["group_id"].astype(str).tolist()
    if (
        len(history_tokens) != len(h_rows)
        or len(set(history_tokens.tolist())) != len(h_rows)
        or set(history_roles.tolist()) != {"H"}
        or dict(zip(history_tokens.tolist(), history_groups.tolist(), strict=True))
        != dict(zip(expected_tokens, expected_groups, strict=True))
    ):
        raise ValueError(f"{fit_id} history tokens/groups are not exactly the H role")

    return {
        "history_receipt_sha256": sha256(history_receipt_path),
        "history_payload_sha256": sha256(history_path),
        "observation_receipt_sha256": sha256(observation_receipt_path),
        "observation_payload_sha256": sha256(observation_path),
        "normalizer_sha256": sha256(normalizer_path),
        "producer_pair_freeze_sha256": freeze_sha,
        "endpoint_hashes": expected_endpoint_hashes,
        "observation_ids_sha256": unique_ids_sha,
        "h_tokens": len(history_tokens),
        "h_groups": len(set(history_groups.tolist())),
        "transitive_closure_verified": True,
    }


def audit_rotation0_reuse(run: Path, role_sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Admit V13 rotation-0 reuse only when every fit and source still binds."""
    try:
        endpoints = {
            fit_id: _audit_endpoint(run / "fits" / fit_id, updates, role)
            for fit_id, (updates, role) in _REUSE_FITS.items()
        }
        p_hash = role_sources["P"]["manifest_sha256"]
        h_hash = role_sources["H"]["manifest_sha256"]
        for fit_id, endpoint in endpoints.items():
            provenance = _json(run / "fits" / fit_id / "RUN_PROVENANCE.json")
            expected = p_hash if endpoint["historical_role"] == "P" else h_hash
            if provenance.get("dataset_manifest_hash") != expected:
                raise ValueError(f"{fit_id} dataset manifest is not its historical role")

        pair_specs = {
            "PAIR_E_MATCHED": ("E_PAIR_TRAIN.json", "E_A5_MATCHED"),
            "PAIR_R": ("R_PAIR_TRAIN.json", "R_A5"),
        }
        sources: dict[str, Any] = {}
        for fit_id, (filename, parent) in pair_specs.items():
            path = run / "features" / "P" / filename
            source = _json(path)
            payload = Path(str(source["output"]))
            if (
                source.get("status") != "COMPLETE"
                or int(source.get("rows", -1)) != 53_890
                or source.get("parent_checkpoint_sha256") != endpoints[parent]["checkpoint_sha256"]
                or source.get("output_sha256") != sha256(payload)
                or _json(run / "fits" / fit_id / "RUN_PROVENANCE.json").get("source_sha256")
                != source.get("output_sha256")
            ):
                raise ValueError(f"{fit_id} source receipt is inconsistent")
            sources[fit_id] = {"receipt_sha256": sha256(path), "payload_sha256": sha256(payload)}

        h_rows_path = Path(str(role_sources["H"]["rows_path"]))
        h_rows = pd.read_parquet(h_rows_path, columns=["sample_token", "group_id"])
        if len(h_rows) != 18_422 or h_rows["sample_token"].astype(str).duplicated().any():
            raise ValueError("Historical H identity rows are incomplete or duplicated")
        for fit_id, prefix, modality in (
            ("E_CTX_MATCHED", "E", "event"),
            ("R_CTX", "R", "rgb"),
        ):
            sources[fit_id] = _audit_context_closure(
                run,
                fit_id=fit_id,
                prefix=prefix,
                modality=modality,
                endpoints=endpoints,
                h_rows=h_rows,
            )
        return {
            "schema": _ROTATION0_AUDIT_SCHEMA,
            "status": "PASS",
            "reuse_admitted": True,
            "fits": endpoints,
            "sources": sources,
            "scope": "Transitive V13 P/H/V ancestry closure for rotation 0 only",
            "checks": {
                "producer_pair_endpoint_freeze": True,
                "observation_endpoint_hashes": True,
                "normalizer_h_only_ids_and_producer": True,
                "history_h_tokens_and_groups": True,
            },
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {
            "schema": _ROTATION0_AUDIT_SCHEMA,
            "status": "NOT_ADMITTED",
            "reuse_admitted": False,
            "reason": f"{type(error).__name__}: {error}",
        }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_fold_contract(
    workspace: Path,
    review_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify and materialize five rotations as small row-reference manifests."""
    workspace = workspace.resolve()
    review_path = review_path.resolve()
    output_dir = output_dir.resolve()
    run = workspace / _V13_RUN
    review = _json(review_path)
    rows, role_sources = load_historical_rows(run)
    validate_rotation_contract(rows, review)
    expected_bindings = {
        str(Path(path).resolve()): digest for path, digest in review["metadata_bindings"].items()
    }
    observed_bindings = {
        str(Path(source[key]).resolve()): source[hash_key]
        for source in role_sources.values()
        for key, hash_key in (("manifest_path", "manifest_sha256"), ("rows_path", "rows_sha256"))
    }
    if expected_bindings != observed_bindings:
        raise ValueError("Review metadata bindings differ from current historical roles")

    output_dir.mkdir(parents=True, exist_ok=True)
    rotation_manifests: list[dict[str, Any]] = []
    for rotation in review["rotations"]:
        role_outputs: dict[str, Any] = {}
        for role in _ROTATION_ROLES:
            selected = select_rotation_rows(rows, rotation, role)
            filename = f"rotation_{int(rotation['rotation'])}_{role}.parquet"
            path = output_dir / filename
            selected.to_parquet(path, index=False)
            role_outputs[role] = {
                "path": filename,
                "sha256": sha256(path),
                "queries": len(selected),
                "groups": int(selected["group_id"].nunique()),
                "columns": list(selected.columns),
                "cache_policy": "REFERENCE_HISTORICAL_ROLE_AND_LOCAL_ORDINAL_NO_PAYLOAD_COPY",
            }
        manifest = {
            "schema": "final_refit_rotation_roles_v1",
            "rotation": int(rotation["rotation"]),
            "status": "VERIFIED_REFERENCE_MANIFESTS",
            "label_columns_read": False,
            "roles": role_outputs,
            "generation_ancestry_excludes_group": True,
        }
        manifest_path = output_dir / f"rotation_{int(rotation['rotation'])}.json"
        _write_json(manifest_path, manifest)
        rotation_manifests.append({"path": manifest_path.name, "sha256": sha256(manifest_path)})

    reuse = audit_rotation0_reuse(run, role_sources)
    reuse_path = output_dir / "ROTATION0_REUSE_AUDIT.json"
    _write_json(reuse_path, reuse)
    result = {
        "schema": "final_refit_oof_fold_contract_v1",
        "status": "PASS",
        "training_started": False,
        "optimizer_updates": 0,
        "targets_read": False,
        "review": {"path": str(review_path), "sha256": sha256(review_path)},
        "historical_sources": role_sources,
        "population": {"queries": len(rows), "groups": int(rows["group_id"].nunique())},
        "rotations": rotation_manifests,
        "generation_coverage": "EACH_TRAIN40_TOKEN_EXACTLY_ONCE",
        "cache_policy": "LOOKUP_ONLY_NO_LARGE_CACHE_DUPLICATION",
        "rotation0_reuse": {
            "status": reuse["status"],
            "admitted": reuse["reuse_admitted"],
            "audit_path": reuse_path.name,
            "audit_sha256": sha256(reuse_path),
        },
        "limits": "Contract and references only; no OOF runner or training is implemented here.",
    }
    contract_path = output_dir / "FOLD_CONTRACT.json"
    _write_json(contract_path, result)
    return result


def iter_cache_lookups(frame: pd.DataFrame) -> Iterable[tuple[str, int]]:
    """Yield stable ``(historical_role, local_ordinal)`` cache addresses."""
    roles = frame["historical_role"].astype(str).tolist()
    ordinals = frame["historical_local_ordinal"].astype(int).tolist()
    yield from zip(roles, ordinals, strict=True)
