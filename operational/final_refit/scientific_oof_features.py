"""Fold-local feature generation: all learned event/RGB columns are recomputed."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.efficient_context.mapped_union import encode_union
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.rgb_port.features import event_statistics_from_channels12
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.infer_experts import clock_features
from operational.rgb_port.train_producers import producer_features

from .event_history import (
    _historical_pair_input,
    _load_sources,
    _query_plan,
    _validate_historical_h8,
    causal_history_mask,
)
from .features import _atomic_npz, _seal_cache, validate_cache_receipt
from .materialize import FragmentStore, materialize_rgb_observations
from .oof_lineage import load_rotation_identity
from .recipe import canonical_sha256, ordered_row_identity_sha256, sha256
from .rgb import AllTrainRGBSource


def arrays(path: Path) -> dict[str, np.ndarray]:
    """Load only non-pickled numerical/string arrays."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def role_rows(contract: Path, rotation: int, role: str) -> pd.DataFrame:
    """Read rows only after group-disjointness and exact manifests are verified."""
    identity = load_rotation_identity(contract, rotation)
    return (
        pd.read_parquet(identity.roles[role].rows_path)
        .sort_values("source_ordinal", kind="stable")
        .reset_index(drop=True)
    )


def physical_phase20_to_state340(physical: np.ndarray) -> np.ndarray:
    """Embed the only consumed coordinates; never mix learned fold hidden states."""
    value = np.asarray(physical)
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != 20:
        raise ValueError("PHASE_ONLY OOF input requires FP32 [N,20]")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite physical phase coordinates")
    output = np.zeros((len(value), 340), dtype=np.float32)
    output[:, 160:170], output[:, 330:340] = value[:, :10], value[:, 10:]
    return output


def normalize_histories(
    raw: Mapping[str, np.ndarray], mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Normalize valid PHASE17 observations only; padding remains exactly zero."""
    valid = np.asarray(raw["valid"], dtype=np.bool_)
    result = np.array(raw["features"], dtype=np.float32, copy=True)
    if (
        result.shape != (*valid.shape, 17)
        or mean.shape != (17,)
        or scale.shape != (17,)
        or np.any(scale <= 0)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(result).all()
    ):
        raise ValueError("Invalid PHASE17 normalization inputs")
    result[valid] = (result[valid] - mean) / scale
    result[~valid] = 0
    return result


def unique_normalizer(
    raw: Mapping[str, np.ndarray],
    *,
    expected_groups: set[str],
    expected_role_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit using unique observations in the declared training groups, never targets."""
    if set(np.asarray(raw["group_id"]).astype(str)) != expected_groups:
        raise ValueError("Normalizer received a different group role")
    role_identity: dict[str, Any] | None = None
    if expected_role_identity is not None:
        tokens = np.asarray(raw.get("sample_token", [])).astype(str)
        ordinals = np.asarray(raw.get("source_ordinal", []), dtype=np.int64)
        groups = np.asarray(raw["group_id"]).astype(str)
        role_identity = {
            "population": len(groups),
            "group_ids": sorted(set(groups.tolist())),
            "sample_token_sha256": canonical_sha256(tokens.tolist()),
            "ordered_native_ordinal_sha256": canonical_sha256(
                [str(value) for value in ordinals.tolist()]
            ),
        }
        expected = {key: expected_role_identity.get(key) for key in role_identity}
        if (
            tokens.shape != groups.shape
            or ordinals.shape != groups.shape
            or role_identity != expected
        ):
            raise ValueError("Normalizer rows differ from the exact fold role identity")
    valid = np.asarray(raw["valid"], dtype=np.bool_)
    ids = np.asarray(raw["observation_id"]).astype(str)[valid]
    values = np.asarray(raw["features"], dtype=np.float32)[valid]
    groups = np.repeat(np.asarray(raw["group_id"]).astype(str)[:, None], valid.shape[1], 1)[valid]
    if len(ids) == 0 or any(not value for value in ids):
        raise ValueError("Normalizer requires nonempty observation identities")
    unique, indices, inverse = np.unique(ids, return_index=True, return_inverse=True)
    # Repeated observations must have the same features and group, not just the same ID.
    if not np.array_equal(values, values[indices][inverse]) or not np.array_equal(
        groups, groups[indices][inverse]
    ):
        raise ValueError("Repeated observation identity has conflicting values or group")
    selected = values[indices].astype(np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite normalizer input")
    scale = selected.std(axis=0)
    result = {
        "mean": selected.mean(axis=0).astype(np.float32).tolist(),
        "scale": np.where(scale > 1e-6, scale, 1.0).astype(np.float32).tolist(),
        "groups": sorted(expected_groups),
        "unique_observations": len(unique),
        "unique_ids_sha256": canonical_sha256(unique.tolist()),
        "targets_read": False,
    }
    if role_identity is not None:
        result["role_identity"] = role_identity
    return result


def _arrays_equal(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        np.asarray(left[key]).dtype == np.asarray(right[key]).dtype
        and np.asarray(left[key]).shape == np.asarray(right[key]).shape
        and np.ascontiguousarray(left[key]).tobytes() == np.ascontiguousarray(right[key]).tobytes()
        for key in left
    )


def _array_content_sha256(values: Mapping[str, np.ndarray]) -> str:
    """Hash array semantics without depending on NPZ ZIP timestamps or compression."""
    digest = hashlib.sha256()
    for key in sorted(values):
        value = np.ascontiguousarray(values[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(canonical_sha256(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _validated_parents(parents: Mapping[str, str]) -> dict[str, str]:
    value = dict(sorted((str(key), str(digest)) for key, digest in parents.items()))
    if not value or any(
        not key
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for key, digest in value.items()
    ):
        raise ValueError("Parents must be named lowercase SHA-256 bindings")
    return value


def _expected_role_receipt(
    path: Path,
    values: Mapping[str, np.ndarray],
    *,
    schema: str,
    role: str,
    parents: Mapping[str, str],
    rows_sha: str,
    details: Mapping[str, Any] | None,
    normalizers: Mapping[str, str] | None,
) -> dict[str, Any]:
    if len(rows_sha) != 64 or any(value not in "0123456789abcdef" for value in rows_sha):
        raise ValueError("Role rows must be bound by a lowercase SHA-256")
    normalizer_bindings = dict(sorted((normalizers or {}).items()))
    if any(
        not name or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest)
        for name, digest in normalizer_bindings.items()
    ):
        raise ValueError("Normalizers must be named lowercase SHA-256 bindings")
    tokens = np.asarray(values["sample_token"]).astype(str).tolist()
    groups = np.asarray(values["group_id"]).astype(str).tolist()
    receipt: dict[str, Any] = {
        "schema": schema,
        "status": "COMPLETE",
        "role": role,
        "rows": len(groups),
        "groups": len(set(groups)),
        "group_ids": sorted(set(groups)),
        "parents": _validated_parents(parents),
        "normalizers": normalizer_bindings,
        "output": str(path.resolve()),
        "output_sha256": sha256(path),
        "compression": "np.savez_compressed",
        "targets_used_for_feature_construction": False,
        "ordered_row_identity_sha256": ordered_row_identity_sha256(tokens, groups),
        "native_train40_rows_sha256": rows_sha,
    }
    if details:
        receipt.update(details)
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    return receipt


def seal_role_cache(
    path: Path,
    values: Mapping[str, np.ndarray],
    *,
    schema: str,
    role: str,
    parents: Mapping[str, str],
    rows_sha: str,
    details: Mapping[str, Any] | None = None,
    normalizers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Seal a role-specific cache using the existing hashed cache format."""
    if set(np.asarray(values["role"]).astype(str)) != {role}:
        raise ValueError("Cache arrays and receipt role differ")
    if path.exists():
        if not _arrays_equal(arrays(path), values):
            raise ValueError("Previously generated cache arrays changed on deterministic replay")
        expected = _expected_role_receipt(
            path,
            values,
            schema=schema,
            role=role,
            parents=parents,
            rows_sha=rows_sha,
            details={**(details or {}), "role": role},
            normalizers=normalizers,
        )
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            atomic_write_json(sidecar, expected)
            return expected
        receipt = validate_cache_receipt(path, schema, expected_role=role)
        if receipt != expected:
            raise ValueError("Previously generated cache receipt or ancestry changed")
        return receipt
    if path.with_suffix(".json").exists():
        raise ValueError("Role cache receipt exists without its bound NPZ")
    return _seal_cache(
        path,
        values,
        schema=schema,
        parents=parents,
        normalizers=normalizers,
        native_train40_rows_sha256=rows_sha,
        details={**(details or {}), "role": role},
    )


def _seal_raw_phase17(
    destination: Path,
    values: Mapping[str, np.ndarray],
    rows: pd.DataFrame,
    *,
    modality: str,
    parents: Mapping[str, str],
    provenance: Mapping[str, Any],
) -> Path:
    _validate_raw_phase17_values(values, rows)
    tokens = rows.sample_token.astype(str).tolist()
    groups = rows.group_id.astype(str).tolist()
    ordinals = rows.source_ordinal.astype(int).tolist()
    if modality not in {"rgb", "event"}:
        raise ValueError("Unknown raw PHASE17 modality")
    if destination.exists():
        if not _arrays_equal(arrays(destination), values):
            raise ValueError("Raw PHASE17 artifact changed on deterministic replay")
    else:
        if destination.with_suffix(".json").exists():
            raise ValueError("Raw PHASE17 receipt exists without its bound NPZ")
        _atomic_npz(destination, values, compressed=True)
    receipt: dict[str, Any] = {
        "schema": "scientific_oof_raw_phase17_v1",
        "status": "COMPLETE",
        "modality": modality,
        "rows": len(rows),
        "ordered_sample_token_sha256": canonical_sha256(tokens),
        "ordered_group_id_sha256": canonical_sha256(groups),
        "ordered_native_ordinal_sha256": canonical_sha256([str(value) for value in ordinals]),
        "parents": _validated_parents(parents),
        "output": str(destination.resolve()),
        "output_sha256": sha256(destination),
        "provenance": _validated_raw_provenance(modality, provenance),
        "targets_read": True,
        "targets_used_as_features": False,
        "targets_used": False,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    sidecar = destination.with_suffix(".json")
    if sidecar.exists():
        if read_json_shared(sidecar) != receipt:
            raise ValueError("Raw PHASE17 receipt or ancestry changed")
    else:
        atomic_write_json(sidecar, receipt)
    return destination


def _validated_raw_provenance(modality: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(sorted(provenance.items()))
    if modality == "rgb":
        digests = {"history_arrays_sha256", "observations_arrays_sha256"}
        valid = set(value) == digests
    elif modality == "event":
        digests = {"historical_h8_sha256"}
        valid = (
            set(value) == {"historical_h8_sha256", "historical_learned_columns_reused"}
            and value.get("historical_learned_columns_reused") == []
        )
    else:
        raise ValueError("Unknown raw PHASE17 modality")
    if not valid or any(
        not isinstance(value.get(key), str)
        or len(value[key]) != 64
        or any(character not in "0123456789abcdef" for character in value[key])
        for key in digests
    ):
        raise ValueError("Raw PHASE17 provenance bindings are invalid")
    return value


def _validate_raw_phase17_values(values: Mapping[str, np.ndarray], rows: pd.DataFrame) -> None:
    required = {
        "features",
        "valid",
        "timing",
        "observation_id",
        "expert_phase",
        "sample_token",
        "group_id",
        "source_ordinal",
    }
    tokens = rows.sample_token.astype(str).tolist()
    groups = rows.group_id.astype(str).tolist()
    ordinals = rows.source_ordinal.astype(int).tolist()
    features = np.asarray(values.get("features", []))
    valid = np.asarray(values.get("valid", []))
    timing = np.asarray(values.get("timing", []))
    observation_id = np.asarray(values.get("observation_id", []))
    expert_phase = np.asarray(values.get("expert_phase", []))
    if (
        set(values) != required
        or len(rows) == 0
        or len(set(tokens)) != len(tokens)
        or len(set(ordinals)) != len(ordinals)
        or np.asarray(values.get("sample_token", [])).astype(str).tolist() != tokens
        or np.asarray(values.get("group_id", [])).astype(str).tolist() != groups
        or np.asarray(values.get("source_ordinal", []), dtype=np.int64).tolist() != ordinals
        or features.dtype != np.float32
        or features.ndim != 3
        or features.shape[0] != len(rows)
        or features.shape[2] != 17
        or valid.dtype != np.bool_
        or valid.shape != features.shape[:2]
        or timing.dtype != np.float32
        or timing.shape != (*valid.shape, 4)
        or observation_id.shape != valid.shape
        or expert_phase.dtype != np.float32
        or expert_phase.shape != (len(rows), 3)
        or not np.isfinite(features).all()
        or not np.isfinite(timing).all()
        or not np.isfinite(expert_phase).all()
        or np.any(features[~valid] != 0)
        or np.any(timing[~valid] != 0)
        or any(not value for value in observation_id[valid].astype(str).tolist())
        or any(value for value in observation_id[~valid].astype(str).tolist())
    ):
        raise ValueError("Raw PHASE17 arrays differ from their exact role rows or layout")


def _reuse_verified_raw_phase17(
    destination: Path,
    rows: pd.DataFrame,
    *,
    modality: str,
    parents: Mapping[str, str],
    expected_provenance: Mapping[str, Any],
) -> bool:
    """Return true only for a physically and semantically verified completed artifact."""
    sidecar = destination.with_suffix(".json")
    if not destination.exists() or not sidecar.exists():
        return False
    values = arrays(destination)
    _validate_raw_phase17_values(values, rows)
    receipt = read_json_shared(sidecar)
    tokens = rows.sample_token.astype(str).tolist()
    groups = rows.group_id.astype(str).tolist()
    ordinals = rows.source_ordinal.astype(int).tolist()
    required = {
        "schema",
        "status",
        "modality",
        "rows",
        "ordered_sample_token_sha256",
        "ordered_group_id_sha256",
        "ordered_native_ordinal_sha256",
        "parents",
        "output",
        "output_sha256",
        "provenance",
        "targets_read",
        "targets_used_as_features",
        "targets_used",
        "receipt_payload_sha256",
    }
    payload = dict(receipt)
    payload_sha = payload.pop("receipt_payload_sha256", None)
    provenance = receipt.get("provenance")
    try:
        validated_provenance = _validated_raw_provenance(
            modality, provenance if isinstance(provenance, dict) else {}
        )
    except ValueError:
        validated_provenance = None
    if (
        set(receipt) != required
        or receipt.get("schema") != "scientific_oof_raw_phase17_v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("modality") != modality
        or receipt.get("rows") != len(rows)
        or receipt.get("ordered_sample_token_sha256") != canonical_sha256(tokens)
        or receipt.get("ordered_group_id_sha256") != canonical_sha256(groups)
        or receipt.get("ordered_native_ordinal_sha256")
        != canonical_sha256([str(value) for value in ordinals])
        or receipt.get("parents") != _validated_parents(parents)
        or receipt.get("output") != str(destination.resolve())
        or receipt.get("output_sha256") != sha256(destination)
        or provenance != validated_provenance
        or provenance != _validated_raw_provenance(modality, expected_provenance)
        or receipt.get("targets_read") is not True
        or receipt.get("targets_used_as_features") is not False
        or receipt.get("targets_used") is not False
        or payload_sha != canonical_sha256(payload)
    ):
        raise ValueError("Raw PHASE17 receipt, ancestry, or physical hash is invalid")
    return True


class RGBRoleView:
    """Inference-only group-filtered view; no fit can see generation/refiner targets."""

    def __init__(self, base: AllTrainRGBSource, rows: pd.DataFrame) -> None:
        self.base, self.rows = base, rows
        self.native = rows.source_ordinal.astype(int).tolist()
        self.population_size = len(rows)
        self.datasets = base.datasets
        for local, native in enumerate(self.native):
            observed = base.row(native)
            row = rows.iloc[local]
            if str(observed["sample_token"]) != str(row.sample_token) or str(
                observed["group_id"]
            ) != str(row.group_id):
                raise ValueError("RGB role source differs from native TRAIN40 identity")

    def row(self, index: int) -> Mapping[str, Any]:
        return self.base.row(self.native[index])

    def history(self, index: int) -> Sequence[Mapping[str, Any]]:
        return self.base.history(self.native[index])


def rgb_raw(
    base: AllTrainRGBSource,
    rows: pd.DataFrame,
    models: Mapping[str, nn.Module],
    output: Path,
    *,
    device: str,
    parents: Mapping[str, str],
    guard: Callable[[], None],
) -> Path:
    """Materialize only this role's causal observations using its rotation's models."""
    destination = output / "raw_phase17.npz"
    observations, history = output / "observations.npz", output / "history.npz"
    if observations.exists() and history.exists():
        if _reuse_verified_raw_phase17(
            destination,
            rows,
            modality="rgb",
            parents=parents,
            expected_provenance={
                "history_arrays_sha256": _array_content_sha256(arrays(history)),
                "observations_arrays_sha256": _array_content_sha256(arrays(observations)),
            },
        ):
            return destination
    if destination.exists():
        # A completed raw artifact is authoritative only after a full deterministic
        # replay.  Isolate that replay from resumable observation/history caches so
        # stale intermediates cannot be used to re-sign an old raw artifact.
        with TemporaryDirectory(prefix=".raw_phase17_replay_", dir=output) as temporary:
            return _rgb_raw_from_paths(
                base,
                rows,
                models,
                Path(temporary),
                destination,
                device=device,
                parents=parents,
                guard=guard,
            )
    return _rgb_raw_from_paths(
        base,
        rows,
        models,
        output,
        destination,
        device=device,
        parents=parents,
        guard=guard,
    )


def _rgb_raw_from_paths(
    base: AllTrainRGBSource,
    rows: pd.DataFrame,
    models: Mapping[str, nn.Module],
    work: Path,
    destination: Path,
    *,
    device: str,
    parents: Mapping[str, str],
    guard: Callable[[], None],
) -> Path:
    observations, history = work / "observations.npz", work / "history.npz"
    materialize_rgb_observations(
        cast(Any, RGBRoleView(base, rows)),
        models["R_A5"],
        models["R_C2F"],
        models["PAIR_R"],
        observations,
        history,
        parents=parents,
        device=device,
        guard=guard,
        include_targets=False,
    )
    obs, hist = arrays(observations), arrays(history)
    row_tokens = rows.sample_token.astype(str).tolist()
    row_groups = rows.group_id.astype(str).tolist()
    valid = np.asarray(hist.get("valid", []), dtype=np.bool_)
    history_ids = np.asarray(hist.get("observation_id", [])).astype(str)
    if (
        len(hist.get("sample_token", [])) != len(rows)
        or np.asarray(hist["sample_token"]).astype(str).tolist() != row_tokens
        or np.asarray(hist["group_id"]).astype(str).tolist() != row_groups
        or valid.ndim != 2
        or valid.shape[0] != len(rows)
        or history_ids.shape != valid.shape
    ):
        raise ValueError("RGB history rows differ from the exact fold role")
    lookup = {str(key): i for i, key in enumerate(obs["observation_id"])}
    if len(lookup) != len(obs["observation_id"]):
        raise ValueError("RGB observations contain duplicate identities")
    raw = np.zeros((*valid.shape, 17), dtype=np.float32)
    for row, column in zip(*np.nonzero(valid), strict=True):
        if str(history_ids[row, column]) not in lookup:
            raise ValueError("RGB history references an unknown observation")
        index = lookup[str(hist["observation_id"][row, column])]
        if str(obs["group_id"][index]) != str(hist["group_id"][row]):
            raise ValueError("RGB history crossed groups")
        query = int(base.row(int(rows.iloc[row].source_ordinal))["query_time_us"])
        anchor, available = int(obs["anchor_us"][index]), int(obs["available_us"][index])
        if (
            anchor > query
            or available > query
            or available < anchor
            or query - anchor + int(obs["input_span_us"][index]) > 650_000
        ):
            raise ValueError("RGB history contains future or over-budget information")
        raw[row, column] = obs["features"][index]
    result = {
        key: hist[key]
        for key in ("observation_id", "timing", "valid", "expert_phase", "sample_token", "group_id")
    }
    result.update(features=raw, source_ordinal=rows.source_ordinal.to_numpy(np.int64))
    return _seal_raw_phase17(
        destination,
        result,
        rows,
        modality="rgb",
        parents=parents,
        provenance={
            "history_arrays_sha256": _array_content_sha256(hist),
            "observations_arrays_sha256": _array_content_sha256(obs),
        },
    )


def event_raw(
    *,
    v12_run: Path,
    raw_root: Path,
    rows: pd.DataFrame,
    models: Mapping[str, nn.Module],
    output: Path,
    parents: Mapping[str, str],
    device: str,
    guard: Callable[[], None],
) -> Path:
    """Regenerate every learned event column; historical H8 supplies indices only."""
    destination = output / "raw_phase17.npz"
    if _reuse_verified_raw_phase17(
        destination,
        rows,
        modality="event",
        parents=parents,
        expected_provenance={
            "historical_h8_sha256": sha256(v12_run / "H8_FEATURES.npz"),
            "historical_learned_columns_reused": [],
        },
    ):
        return destination
    sources = _load_sources(v12_run, raw_root)
    _, history_indices, _, _ = _validate_historical_h8(v12_run)
    store = FragmentStore(
        output / "parts",
        {
            "schema": "scientific_oof_event_history_v1",
            "ordered_tokens": canonical_sha256(rows.sample_token.astype(str).tolist()),
            "parents": dict(parents),
            "old_history_indices_sha256": sha256(v12_run / "H8_FEATURES.npz"),
            "historical_learned_columns_reused": [],
            "device": device,
        },
    )
    pool = ReaderPool()
    parts = []
    target = torch.device(device)
    deterministic_replay = destination.exists()
    try:
        with torch.inference_mode():
            for offset, row in enumerate(rows.to_dict("records")):
                saved = None if deterministic_replay else store.read(offset)
                if saved is not None:
                    parts.append(saved)
                    continue
                guard()
                native = int(row["source_ordinal"])
                if str(sources.index["tokens"][native]) != str(row["sample_token"]) or str(
                    sources.index["sequences"][native]
                ) != str(row["group_id"]):
                    raise ValueError("Event role differs from native TRAIN40 row")
                plan = _query_plan(sources, raw_root, history_indices, native)
                sensor = encode_union(
                    pool.get(plan.path),
                    plan.windows,
                    np.arange(15, -1, -1, dtype=np.int64) * 50_000,
                    np.concatenate((np.zeros(8, dtype=np.bool_), plan.valid)),
                    tuple(float(x) for x in plan.square),
                    sequence_id=plan.sequence,
                    roi_size=128,
                    event_pixel_diff=5,
                ).to(target, dtype=torch.float32)
                delta = torch.full((16, 2), plan.delta_seconds, dtype=torch.float32, device=target)
                outputs = [
                    producer_features(cast(CausalScaleTTC, models[name]), sensor, delta)
                    for name in ("E_A5", "E_C2F")
                ]
                pair = emitted_phase(
                    models["PAIR_E"](
                        PairFeatureBatch(_historical_pair_input(outputs[0], delta[:, -1]))
                    )
                )
                experts = torch.stack(
                    (outputs[0]["point_phase"], outputs[1]["point_phase"], pair), -1
                )
                diagnostics = [
                    torch.stack((x["flow"], x["margin"], x["log_variance"]), -1) for x in outputs
                ]
                signed = torch.stack(
                    (
                        experts[:, 2] - experts[:, 0],
                        experts[:, 2] - experts[:, 1],
                        experts[:, 1] - experts[:, 0],
                    ),
                    -1,
                )
                physical = (
                    torch.cat(
                        (
                            event_statistics_from_channels12(sensor[:, -1]),
                            *diagnostics,
                            experts,
                            signed,
                            signed.abs(),
                        ),
                        -1,
                    )
                    .cpu()
                    .numpy()[-8:]
                )
                columns = np.flatnonzero(plan.valid)
                anchors = [plan.anchor_us - (7 - int(c)) * 50_000 for c in columns]
                causal = causal_history_mask(
                    anchors, query_us=plan.available_us, input_span_us=plan.input_span_us
                )
                columns = columns[causal]
                anchors = np.asarray(anchors)[causal].tolist()
                valid = np.zeros(8, dtype=np.bool_)
                valid[columns] = True
                features = np.zeros((8, 17), dtype=np.float32)
                features[columns] = physical[columns]
                timing = np.zeros((8, 4), dtype=np.float32)
                timing[columns] = clock_features(
                    anchors, [plan.available_us] * len(anchors), plan.available_us
                )
                ids = np.full(8, "", dtype="U64")
                ids[columns] = np.asarray(plan.observation_ids)[causal]
                value = {
                    "features": features[None],
                    "valid": valid[None],
                    "timing": timing[None],
                    "observation_id": ids[None],
                    "expert_phase": physical[-1, 8:11][None],
                    "sample_token": np.asarray([str(row["sample_token"])]),
                    "group_id": np.asarray([str(row["group_id"])]),
                    "source_ordinal": np.asarray([native], dtype=np.int64),
                }
                if not deterministic_replay:
                    store.write(offset, value)
                parts.append(value)
    finally:
        pool.close()
    if not parts:
        raise ValueError("Event role contains no rows")
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    return _seal_raw_phase17(
        destination,
        result,
        rows,
        modality="event",
        parents=parents,
        provenance={
            "historical_h8_sha256": sha256(v12_run / "H8_FEATURES.npz"),
            "historical_learned_columns_reused": [],
        },
    )
