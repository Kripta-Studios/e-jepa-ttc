"""Lossless, ancestry-bound feature caches for the final TRAIN40 refit.

The builders in this module intentionally do not own model training.  They seal the
interfaces between the frozen RGB producers, PAIR_R, both temporal refiners and the
V16 PHASE_ONLY readout.  Targets are copied only into supervised training caches;
they are never consulted while fitting normalizers or assembling histories.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.rgb_port.features import (
    EVENT_PHASE17_NAMES,
    PAIR_FEATURE_NAMES,
    schema_sha256,
)
from e_jepa_ttc.rgb_port.features import (
    RGB_PHASE17_SHA256 as HISTORICAL_RGB_PHASE17_SHA256,
)
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase

from .recipe import ordered_row_identity_sha256

ROLE = "TRAIN"
BLOCK_WIDTH = 170
HISTORY_LIMIT = 8
PAIR_WIDTH = 133
PHASE_WIDTH = 340

PAIR_RGB_SCHEMA_SHA256 = schema_sha256(PAIR_FEATURE_NAMES, modality="rgb", version=1)
PAIR_EVENT_SCHEMA_SHA256 = schema_sha256(PAIR_FEATURE_NAMES, modality="event", version=1)
# V13/V16 used the normalizer module's original names-only RGB hash.  Preserve it
# byte-for-byte instead of silently "improving" the identity during final refit.
RGB_PHASE17_SCHEMA_SHA256 = HISTORICAL_RGB_PHASE17_SHA256
EVENT_PHASE17_SCHEMA_SHA256 = schema_sha256(EVENT_PHASE17_NAMES, modality="event", version=1)
PHASE340_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "block": [
                "hidden160",
                "raw_location",
                "emitted_phase",
                "expert_phase3",
                "current_timing4",
                "valid_count_div_8",
            ],
            "modalities": ["rgb", "event"],
            "order": "rgb_then_event",
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def observation_identity_sha256(
    observation_ids: np.ndarray, group_ids: np.ndarray, roles: np.ndarray
) -> str:
    """Hash an observation set canonically; row order cannot change its identity."""

    rows = len(observation_ids)
    identities = _strings(observation_ids, name="observation_id", rows=rows)
    groups = _strings(group_ids, name="group_id", rows=rows)
    role = _strings(roles, name="role", rows=rows)
    return _canonical_sha256(
        sorted([identities[index], groups[index], role[index]] for index in range(rows))
    )


def normalizer_population_binding(
    observations: Mapping[str, np.ndarray],
    history: Mapping[str, np.ndarray],
    *,
    native_train40_rows_sha256: str,
    observation_receipt_sha256: str,
) -> dict[str, Any]:
    """Describe the externally expected TRAIN40 query and observation inventories."""

    rows = len(history["sample_token"])
    roles = _strings(history["role"], name="role", rows=rows)
    groups = _strings(history["group_id"], name="group_id", rows=rows)
    observation_rows = len(observations["observation_id"])
    observation_roles = _strings(
        observations["role"], name="observation role", rows=observation_rows
    )
    observation_groups = _strings(
        observations["group_id"], name="observation group_id", rows=observation_rows
    )
    requested = np.asarray(history["observation_id"]).astype(str)
    valid = np.asarray(history["valid"], dtype=np.bool_)
    if requested.ndim != 2 or requested.shape != valid.shape or not np.any(valid):
        raise ValueError("normalizer history observation IDs and masks differ")
    referenced = requested[valid]
    if any(not identity for identity in referenced.tolist()):
        raise ValueError("normalizer history contains an empty valid observation")
    observation_ids = _strings(
        observations["observation_id"], name="observation_id", rows=observation_rows
    )
    if set(observation_ids.tolist()) != set(referenced.tolist()):
        raise ValueError("normalizer observations differ from the exact referenced history set")
    observation_group = dict(
        zip(observation_ids.tolist(), observation_groups.tolist(), strict=True)
    )
    for row, column in zip(*np.nonzero(valid), strict=True):
        identity = requested[row, column]
        if observation_group[identity] != groups[row]:
            raise ValueError("normalizer history observation crosses a TRAIN group boundary")
    for name, digest in {
        "native_train40_rows_sha256": native_train40_rows_sha256,
        "observation_receipt_sha256": observation_receipt_sha256,
    }.items():
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if set(roles.tolist()) != {ROLE} or set(observation_roles.tolist()) != {ROLE}:
        raise ValueError("normalizer population binding must contain only TRAIN rows")
    return {
        "schema": "final_refit_normalizer_population_v1",
        "role": ROLE,
        "rows": rows,
        "group_ids": sorted(set(groups.tolist())),
        "ordered_row_identity_sha256": ordered_row_identity_sha256(
            np.asarray(history["sample_token"]).astype(str).tolist(), groups.tolist()
        ),
        "native_train40_rows_sha256": native_train40_rows_sha256,
        "observation_rows": observation_rows,
        "observation_group_ids": sorted(set(observation_groups.tolist())),
        "ordered_observation_identity_sha256": observation_identity_sha256(
            observation_ids, observation_groups, observation_roles
        ),
        "observation_receipt_sha256": observation_receipt_sha256,
    }


def _validate_normalizer_population_binding(
    observations: Mapping[str, np.ndarray], binding: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schema",
        "role",
        "rows",
        "group_ids",
        "ordered_row_identity_sha256",
        "native_train40_rows_sha256",
        "observation_rows",
        "observation_group_ids",
        "ordered_observation_identity_sha256",
        "observation_receipt_sha256",
    }
    if set(binding) != required:
        difference = sorted(set(binding) ^ required)
        raise ValueError(f"normalizer population binding fields differ: {difference}")
    result = dict(binding)
    observation_rows = len(observations["observation_id"])
    observation_groups = _strings(
        observations["group_id"], name="observation group_id", rows=observation_rows
    )
    observation_roles = _strings(
        observations["role"], name="observation role", rows=observation_rows
    )
    digests = (
        "ordered_row_identity_sha256",
        "native_train40_rows_sha256",
        "ordered_observation_identity_sha256",
        "observation_receipt_sha256",
    )
    if (
        result["schema"] != "final_refit_normalizer_population_v1"
        or result["role"] != ROLE
        or int(result["rows"]) <= 0
        or int(result["observation_rows"]) != observation_rows
        or sorted(str(item) for item in result["observation_group_ids"])
        != sorted(set(observation_groups.tolist()))
        or set(observation_roles.tolist()) != {ROLE}
        or result["ordered_observation_identity_sha256"]
        != observation_identity_sha256(
            observations["observation_id"], observation_groups, observation_roles
        )
        or any(
            not isinstance(result[name], str)
            or len(result[name]) != 64
            or any(char not in "0123456789abcdef" for char in result[name])
            for name in digests
        )
    ):
        raise ValueError("normalizer population differs from its sealed external inventory")
    return result


def _strings(value: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    result = np.asarray(value).astype(str)
    if result.shape != (rows,) or any(not item for item in result.tolist()):
        raise ValueError(f"{name} must contain {rows} nonempty strings")
    return result


def _float32(value: np.ndarray, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite float32 with shape {shape}")
    return np.ascontiguousarray(result)


def _validate_parents(parents: Mapping[str, str]) -> dict[str, str]:
    result = dict(sorted((str(name), str(digest)) for name, digest in parents.items()))
    if not result:
        raise ValueError("at least one frozen parent binding is required")
    if any(
        not name or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
        for name, digest in result.items()
    ):
        raise ValueError("parent bindings must map names to lowercase SHA-256 digests")
    return result


def bind_files(paths: Mapping[str, Path]) -> dict[str, str]:
    """Resolve and hash the exact immutable files used by a cache stage."""

    if not paths:
        raise ValueError("at least one parent file is required")
    return _validate_parents(
        {name: sha256_file(Path(path).resolve(strict=True)) for name, path in paths.items()}
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    with temporary.open("xb") as stream:
        saver = cast(Any, np.savez_compressed if compressed else np.savez)
        saver(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _seal_cache(
    output: Path,
    values: Mapping[str, np.ndarray],
    *,
    schema: str,
    parents: Mapping[str, str],
    normalizers: Mapping[str, str] | None = None,
    compressed: bool = True,
    details: Mapping[str, Any] | None = None,
    native_train40_rows_sha256: str | None = None,
) -> dict[str, Any]:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace sealed cache {output}")
    validated_parents = _validate_parents(parents)
    group_ids = sorted(set(np.asarray(values["group_id"]).astype(str).tolist()))
    ordered_identity: str | None = None
    if "sample_token" in values:
        if (
            native_train40_rows_sha256 is None
            or len(native_train40_rows_sha256) != 64
            or any(char not in "0123456789abcdef" for char in native_train40_rows_sha256)
        ):
            raise ValueError("TRAIN query caches require the native TRAIN40 rows SHA-256")
        ordered_identity = ordered_row_identity_sha256(
            np.asarray(values["sample_token"]).astype(str).tolist(),
            groups=np.asarray(values["group_id"]).astype(str).tolist(),
        )
    reserved = {"ordered_row_identity_sha256", "native_train40_rows_sha256"}
    if details and reserved.intersection(details):
        raise ValueError("cache details cannot override the frozen TRAIN40 population")
    _atomic_npz(output, values, compressed=compressed)
    receipt: dict[str, Any] = {
        "schema": schema,
        "status": "COMPLETE",
        "role": ROLE,
        "rows": len(values["group_id"]),
        "groups": len(group_ids),
        "group_ids": group_ids,
        "parents": validated_parents,
        "normalizers": dict(sorted((normalizers or {}).items())),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "compression": "np.savez_compressed" if compressed else "np.savez",
        "targets_used_for_feature_construction": False,
    }
    if ordered_identity is not None:
        receipt["ordered_row_identity_sha256"] = ordered_identity
        receipt["native_train40_rows_sha256"] = native_train40_rows_sha256
    if details:
        receipt.update(details)
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    _atomic_json(output.with_suffix(".json"), receipt)
    return receipt


def load_sealed_cache(path: Path, *, schema: str, exact_keys: set[str]) -> dict[str, np.ndarray]:
    """Reload a cache only after checking its receipt and full file hash."""

    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != schema
        or receipt.get("status") != "COMPLETE"
        or receipt.get("role") != ROLE
        or receipt.get("output_sha256") != sha256_file(path)
    ):
        raise ValueError("sealed cache identity or contents changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != exact_keys:
            raise ValueError("sealed cache fields differ from its stage schema")
        return {name: archive[name] for name in archive.files}


def validate_cache_receipt(
    cache_path: Path, expected_schema: str, *, expected_role: str = ROLE
) -> dict[str, Any]:
    """Validate the common sidecar contract before a trainer opens any arrays."""

    receipt_path = cache_path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "status",
        "role",
        "rows",
        "groups",
        "group_ids",
        "parents",
        "normalizers",
        "output",
        "output_sha256",
        "targets_used_for_feature_construction",
        "receipt_payload_sha256",
    }
    if not required.issubset(receipt):
        raise ValueError(f"cache receipt lacks fields: {sorted(required - set(receipt))}")
    payload = dict(receipt)
    claimed_payload_sha = payload.pop("receipt_payload_sha256")
    if (
        receipt["schema"] != expected_schema
        or receipt["status"] != "COMPLETE"
        or receipt["role"] != expected_role
        or receipt["rows"] <= 0
        or receipt["groups"] != len(receipt["group_ids"])
        or receipt["targets_used_for_feature_construction"] is not False
        or receipt["output"] != str(cache_path.resolve())
        or receipt["output_sha256"] != sha256_file(cache_path)
        or claimed_payload_sha != _canonical_sha256(payload)
    ):
        raise ValueError("cache receipt identity, ancestry or payload changed")
    _validate_parents(receipt["parents"])
    return receipt


def build_pair_cache(
    source: Mapping[str, np.ndarray],
    output: Path,
    *,
    modality: str,
    parents: Mapping[str, str],
    native_train40_rows_sha256: str,
    compressed: bool = True,
) -> dict[str, Any]:
    """Build exact PAIR_E/PAIR_R inputs from the modality-local frozen A5 pass."""

    required = {
        "token128",
        "delta_t_s",
        "support",
        "target_phase",
        "sequence_id",
        "sample_token",
        "group_id",
        "role",
    }
    if set(source) != required or modality not in {"event", "rgb"}:
        raise ValueError(f"PAIR_R source fields differ: {sorted(set(source) ^ required)}")
    rows = len(source["target_phase"])
    role = _strings(source["role"], name="role", rows=rows)
    if set(role.tolist()) != {ROLE}:
        raise ValueError("PAIR_R cache must be built only from TRAIN rows")
    token = _float32(source["token128"], (rows, 128), name="token128")
    delta = np.asarray(source["delta_t_s"], dtype=np.float32)
    if delta.shape == (rows, 2):
        delta = delta[:, -1]
    delta = _float32(delta, (rows,), name="delta_t_s")
    support = np.asarray(source["support"], dtype=np.float32)
    if support.shape not in {(rows, 2), (rows, 3)} or not np.isfinite(support).all():
        raise ValueError("support must be finite [N,2] or [N,3]")
    if np.any(delta <= 0) or np.any((support < 0) | (support > 1)):
        raise ValueError("PAIR_R delta/support values are outside their contract")
    current = support[:, -1]
    features = np.concatenate(
        (
            token,
            delta[:, None],
            np.log(delta)[:, None],
            np.reciprocal(delta)[:, None],
            current[:, None],
            np.minimum(support[:, -2], current)[:, None],
        ),
        axis=1,
        dtype=np.float32,
    )
    values = {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "target_phase": _float32(source["target_phase"], (rows,), name="target_phase"),
        "group_id": _strings(source["group_id"], name="group_id", rows=rows),
        "sample_token": _strings(source["sample_token"], name="sample_token", rows=rows),
        "sequence_id": _strings(source["sequence_id"], name="sequence_id", rows=rows),
        "role": role,
    }
    if len(set(values["sample_token"].tolist())) != rows:
        raise ValueError("PAIR_R sample_token values must be unique")
    return _seal_cache(
        output,
        values,
        schema="final_refit_pair_cache_v1",
        parents=parents,
        compressed=compressed,
        native_train40_rows_sha256=native_train40_rows_sha256,
        details={
            "modality": modality,
            "feature_schema_sha256": (
                PAIR_RGB_SCHEMA_SHA256 if modality == "rgb" else PAIR_EVENT_SCHEMA_SHA256
            ),
            "feature_width": PAIR_WIDTH,
            "phase_contract": "emitted/clamped +/-60 seconds",
        },
    )


def build_pair_cache_from_features(
    features: np.ndarray,
    supervision: Mapping[str, np.ndarray],
    output: Path,
    *,
    modality: str,
    parents: Mapping[str, str],
    native_train40_rows_sha256: str,
    compressed: bool = True,
) -> dict[str, Any]:
    """Seal an already materialized canonical 133-D cache after full validation."""

    required = {"target_phase", "group_id", "sample_token", "sequence_id", "role"}
    if set(supervision) != required or modality not in {"event", "rgb"}:
        raise ValueError("precomputed PAIR supervision fields or modality differ")
    rows = len(supervision["target_phase"])
    values = {
        "features": _float32(features, (rows, PAIR_WIDTH), name="features"),
        "target_phase": _float32(supervision["target_phase"], (rows,), name="target_phase"),
        "group_id": _strings(supervision["group_id"], name="group_id", rows=rows),
        "sample_token": _strings(supervision["sample_token"], name="sample_token", rows=rows),
        "sequence_id": _strings(supervision["sequence_id"], name="sequence_id", rows=rows),
        "role": _strings(supervision["role"], name="role", rows=rows),
    }
    if set(values["role"].tolist()) != {ROLE} or len(set(values["sample_token"])) != rows:
        raise ValueError("precomputed PAIR cache must contain unique TRAIN rows")
    return _seal_cache(
        output,
        values,
        schema="final_refit_pair_cache_v1",
        parents=parents,
        compressed=compressed,
        native_train40_rows_sha256=native_train40_rows_sha256,
        details={
            "modality": modality,
            "feature_schema_sha256": (
                PAIR_RGB_SCHEMA_SHA256 if modality == "rgb" else PAIR_EVENT_SCHEMA_SHA256
            ),
            "feature_width": PAIR_WIDTH,
            "phase_contract": "emitted/clamped +/-60 seconds",
            "precomputed_feature_bytes_validated": True,
        },
    )


def build_observation_cache(
    source: Mapping[str, np.ndarray],
    output: Path,
    *,
    modality: str,
    parents: Mapping[str, str],
    compressed: bool = True,
) -> dict[str, Any]:
    """Assemble one raw PHASE17 row per unique frozen observation."""

    required = {
        "observation_id",
        "base_statistics2",
        "a5_diagnostics3",
        "c2f_diagnostics3",
        "expert_phase",
        "anchor_us",
        "available_us",
        "input_span_us",
        "group_id",
        "role",
    }
    if set(source) != required or modality not in {"rgb", "event"}:
        raise ValueError("observation source fields or modality differ")
    rows = len(source["observation_id"])
    ids = _strings(source["observation_id"], name="observation_id", rows=rows)
    if len(set(ids.tolist())) != rows:
        raise ValueError("normalizer observations must be unique")
    role = _strings(source["role"], name="role", rows=rows)
    if set(role.tolist()) != {ROLE}:
        raise ValueError("observations must belong only to TRAIN")
    base = _float32(source["base_statistics2"], (rows, 2), name="base_statistics2")
    a5 = _float32(source["a5_diagnostics3"], (rows, 3), name="a5_diagnostics3")
    c2f = _float32(source["c2f_diagnostics3"], (rows, 3), name="c2f_diagnostics3")
    expert = _float32(source["expert_phase"], (rows, 3), name="expert_phase")
    signed = np.stack(
        (expert[:, 2] - expert[:, 0], expert[:, 2] - expert[:, 1], expert[:, 1] - expert[:, 0]),
        axis=-1,
    )
    features = np.concatenate((base, a5, c2f, expert, signed, np.abs(signed)), axis=-1)
    anchors = np.asarray(source["anchor_us"], dtype=np.int64)
    available = np.asarray(source["available_us"], dtype=np.int64)
    input_span = np.asarray(source["input_span_us"], dtype=np.int64)
    if (
        anchors.shape != (rows,)
        or available.shape != (rows,)
        or input_span.shape != (rows,)
        or np.any(available < anchors)
        or np.any(input_span <= 0)
        or np.any(input_span > 650_000)
    ):
        raise ValueError("observation clocks are invalid or noncausal")
    schema_sha = RGB_PHASE17_SCHEMA_SHA256 if modality == "rgb" else EVENT_PHASE17_SCHEMA_SHA256
    values = {
        "observation_id": ids,
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "anchor_us": anchors,
        "available_us": available,
        "input_span_us": input_span,
        "group_id": _strings(source["group_id"], name="group_id", rows=rows),
        "role": role,
    }
    return _seal_cache(
        output,
        values,
        schema=f"final_refit_{modality}_observations_v1",
        parents=parents,
        compressed=compressed,
        details={"feature_schema_sha256": schema_sha, "feature_width": 17},
    )


def fit_train40_normalizer(
    observations: Mapping[str, np.ndarray],
    output: Path,
    *,
    modality: str,
    parents: Mapping[str, str],
    population_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit mean/std from every unique TRAIN40 observation and seal its ancestry."""

    if modality not in {"rgb", "event"}:
        raise ValueError("normalizer modality must be rgb or event")
    rows = len(observations["observation_id"])
    ids = _strings(observations["observation_id"], name="observation_id", rows=rows)
    role = _strings(observations["role"], name="role", rows=rows)
    if set(role.tolist()) != {ROLE} or len(set(ids.tolist())) != rows:
        raise ValueError("normalizer fit requires unique TRAIN40 observations")
    features = np.asarray(observations["features"], dtype=np.float64)
    if features.shape != (rows, 17) or not np.isfinite(features).all():
        raise ValueError("normalizer features must be finite [N,17]")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    population = _validate_normalizer_population_binding(observations, population_binding)
    ordered_ids = sorted(ids.tolist())
    payload: dict[str, Any] = {
        "schema": "final_refit_train40_normalizer_v1",
        "status": "COMPLETE",
        "role": ROLE,
        "modality": modality,
        "rows": rows,
        "unique_observations": rows,
        "unique_ids_sha256": hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest(),
        "feature_schema_sha256": (
            RGB_PHASE17_SCHEMA_SHA256 if modality == "rgb" else EVENT_PHASE17_SCHEMA_SHA256
        ),
        "parents": _validate_parents(parents),
        "population": population,
        "mean": mean.astype(np.float32).tolist(),
        "scale": scale.astype(np.float32).tolist(),
        "targets_read": False,
    }
    payload["normalizer_payload_sha256"] = _canonical_sha256(payload)
    if output.exists():
        raise FileExistsError(f"refusing to replace frozen normalizer {output}")
    _atomic_json(output, payload)
    return {**payload, "output": str(output.resolve()), "output_sha256": sha256_file(output)}


def _load_normalizer(
    path: Path,
    *,
    modality: str,
    expected_parents: Mapping[str, str],
    expected_population: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed_payload_sha = payload.get("normalizer_payload_sha256")
    unhashed = dict(payload)
    unhashed.pop("normalizer_payload_sha256", None)
    schema = RGB_PHASE17_SCHEMA_SHA256 if modality == "rgb" else EVENT_PHASE17_SCHEMA_SHA256
    if (
        payload.get("schema") != "final_refit_train40_normalizer_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("role") != ROLE
        or payload.get("modality") != modality
        or payload.get("feature_schema_sha256") != schema
        or payload.get("parents") != _validate_parents(expected_parents)
        or payload.get("population") != dict(expected_population)
        or payload.get("targets_read") is not False
        or claimed_payload_sha != _canonical_sha256(unhashed)
    ):
        raise ValueError("normalizer identity differs from the frozen context ancestry")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if mean.shape != (17,) or scale.shape != (17,) or np.any(scale <= 0):
        raise ValueError("normalizer statistics differ from PHASE17")
    return mean, scale, sha256_file(path)


def build_context_cache(
    observations: Mapping[str, np.ndarray],
    history: Mapping[str, np.ndarray],
    normalizer_path: Path,
    output: Path,
    *,
    modality: str,
    parents: Mapping[str, str],
    population_binding: Mapping[str, Any],
    compressed: bool = True,
) -> dict[str, Any]:
    """Gather the unchanged <=650 ms causal history and normalize valid rows."""

    required = {
        "observation_id",
        "timing",
        "valid",
        "expert_phase",
        "target_phase",
        "sample_token",
        "group_id",
        "role",
    }
    if set(history) != required:
        raise ValueError(f"history fields differ: {sorted(set(history) ^ required)}")
    rows = len(history["sample_token"])
    requested = np.asarray(history["observation_id"]).astype(str)
    valid = np.asarray(history["valid"], dtype=np.bool_)
    if requested.ndim != 2 or requested.shape != valid.shape:
        raise ValueError("history observation IDs and masks differ")
    length = requested.shape[1]
    if length < 1 or length > HISTORY_LIMIT or not np.all(valid[:, -1]):
        raise ValueError("history must contain a valid current observation and at most 8 slots")
    if np.any(valid[:, :-1] & ~valid[:, 1:]):
        raise ValueError("history valid masks must be suffix-valid")
    role = _strings(history["role"], name="role", rows=rows)
    if set(role.tolist()) != {ROLE}:
        raise ValueError("context cache must contain only TRAIN rows")
    population = _validate_normalizer_population_binding(observations, population_binding)
    history_groups = np.asarray(history["group_id"]).astype(str)
    if (
        population["rows"] != rows
        or population["group_ids"] != sorted(set(history_groups.tolist()))
        or population["ordered_row_identity_sha256"]
        != ordered_row_identity_sha256(
            np.asarray(history["sample_token"]).astype(str).tolist(), history_groups.tolist()
        )
    ):
        raise ValueError("context queries differ from the sealed TRAIN40 population")
    ids = np.asarray(observations["observation_id"]).astype(str)
    lookup = {identity: index for index, identity in enumerate(ids.tolist())}
    if len(lookup) != len(ids):
        raise ValueError("observation cache identities must be unique")
    raw = np.zeros((rows, length, 17), dtype=np.float32)
    query_groups = _strings(history["group_id"], name="group_id", rows=rows)
    observation_groups = np.asarray(observations["group_id"]).astype(str)
    anchors = np.asarray(observations["anchor_us"], dtype=np.int64)
    available = np.asarray(observations["available_us"], dtype=np.int64)
    input_span = np.asarray(observations["input_span_us"], dtype=np.int64)
    timing = _float32(history["timing"], (rows, length, 4), name="timing")
    for row, column in zip(*np.nonzero(valid), strict=True):
        identity = requested[row, column]
        if identity not in lookup:
            raise ValueError(f"history references unknown observation {identity}")
        observation = lookup[identity]
        if observation_groups[observation] != query_groups[row]:
            raise ValueError("history observation crosses a TRAIN group boundary")
        raw[row, column] = observations["features"][observation]
    # Reconstruct integer query clocks from every valid observation.  The four
    # historical timing fields are [q-anchor, q-available, delta-anchor,
    # available-anchor] in seconds.  FP32 conversion permits at most 2 us drift.
    tolerance_us = 2.0
    for row in range(rows):
        columns = np.flatnonzero(valid[row])
        observation_indices = [lookup[requested[row, column]] for column in columns]
        inferred_query = np.asarray(
            [
                anchors[index] + float(timing[row, column, 0]) * 1_000_000
                for column, index in zip(columns, observation_indices, strict=True)
            ]
        )
        query_us = float(inferred_query[-1])
        if np.max(np.abs(inferred_query - query_us)) > tolerance_us:
            raise ValueError("history observations imply inconsistent query clocks")
        previous_anchor: int | None = None
        for column, index in zip(columns, observation_indices, strict=True):
            anchor = int(anchors[index])
            arrival = int(available[index])
            span = int(input_span[index])
            expected = np.asarray(
                [
                    query_us - anchor,
                    query_us - arrival,
                    0 if previous_anchor is None else anchor - previous_anchor,
                    arrival - anchor,
                ],
                dtype=np.float64,
            )
            observed_us = timing[row, column].astype(np.float64) * 1_000_000
            if anchor > query_us or arrival > query_us or arrival < anchor:
                raise ValueError("history contains an observation unavailable at query time")
            if query_us - anchor + span > 650_000 + tolerance_us:
                raise ValueError("history exceeds the original total 650 ms information budget")
            if np.max(np.abs(observed_us - expected)) > tolerance_us:
                raise ValueError("history timing fields differ from sealed integer clocks")
            previous_anchor = anchor
    mean, scale, normalizer_sha = _load_normalizer(
        normalizer_path,
        modality=modality,
        expected_parents=parents,
        expected_population=population,
    )
    raw[valid] = (raw[valid] - mean) / scale
    if np.any(timing[valid] < 0):
        raise ValueError("causal timing coordinates must be nonnegative")
    values = {
        "features": raw,
        "timing": timing,
        "valid": valid,
        "expert_phase": _float32(history["expert_phase"], (rows, 3), name="expert_phase"),
        "target_phase": _float32(history["target_phase"], (rows,), name="target_phase"),
        "group_id": query_groups,
        "sample_token": _strings(history["sample_token"], name="sample_token", rows=rows),
        "role": role,
    }
    if len(set(values["sample_token"].tolist())) != rows:
        raise ValueError("context sample_token values must be unique")
    return _seal_cache(
        output,
        values,
        schema="final_refit_context_cache_v1",
        parents=parents,
        normalizers={modality: normalizer_sha},
        compressed=compressed,
        native_train40_rows_sha256=population["native_train40_rows_sha256"],
        details={
            "feature_schema_sha256": (
                RGB_PHASE17_SCHEMA_SHA256 if modality == "rgb" else EVENT_PHASE17_SCHEMA_SHA256
            ),
            "modality": modality,
            "history_slots": length,
            "history_contract": "original causal <=650ms history; suffix-valid; current required",
            "causal_history_revalidated_from_integer_observation_clocks": True,
            "normalizer_population_sha256": _canonical_sha256(population),
        },
    )


def frozen_context_state(
    model: nn.Module,
    features: Tensor,
    timing: Tensor,
    valid: Tensor,
    experts: Tensor,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Extract V16's exact 170 coordinates and native width logits."""

    captured: dict[str, Tensor] = {}

    def hidden_hook(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        del module
        captured["hidden"] = inputs[0].detach().clone()

    def width_hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        del module, inputs
        captured["width_logits"] = output.detach().clone()

    hooks = [
        model.get_submodule("location").register_forward_pre_hook(hidden_hook),
        model.get_submodule("width").register_forward_hook(width_hook),
    ]
    try:
        with torch.inference_mode():
            output = model(features, timing, valid, experts)
    finally:
        for hook in hooks:
            hook.remove()
    block = torch.cat(
        (
            captured["hidden"],
            output["raw_location"][:, None],
            output["point_phase"][:, None],
            experts,
            timing[:, -1],
            valid.sum(dim=1, keepdim=True).to(features.dtype) / HISTORY_LIMIT,
        ),
        dim=-1,
    )
    if block.shape != (len(features), BLOCK_WIDTH) or captured["width_logits"].shape != (
        len(features),
        2,
    ):
        raise ValueError("context model does not expose the V16 frozen-state contract")
    return block, captured["width_logits"], output


def _tensor_batch(
    values: Mapping[str, np.ndarray], names: Sequence[str], rows: slice, device: str
) -> list[Tensor]:
    return [torch.from_numpy(values[name][rows]).to(device) for name in names]


def build_phase_cache(
    event: Mapping[str, np.ndarray],
    rgb: Mapping[str, np.ndarray],
    event_model: nn.Module,
    rgb_model: nn.Module,
    output: Path,
    *,
    parents: Mapping[str, str],
    normalizers: Mapping[str, str],
    native_train40_rows_sha256: str,
    device: str = "cpu",
    batch_size: int = 128,
    compressed: bool = True,
) -> dict[str, Any]:
    """Extract [R170,E170] states and verify the zero readout is exactly R_CTX."""

    if batch_size <= 0 or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("invalid phase-cache execution device or batch size")
    required = {
        "features",
        "timing",
        "valid",
        "expert_phase",
        "target_phase",
        "group_id",
        "sample_token",
        "role",
    }
    if set(event) != required or set(rgb) != required:
        raise ValueError("phase inputs must be exact final-refit context caches")
    rows = len(event["sample_token"])
    for name in ("sample_token", "group_id", "target_phase", "role"):
        if not np.array_equal(event[name], rgb[name]):
            raise ValueError(f"event/RGB context alignment differs: {name}")
    if set(np.asarray(event["role"]).astype(str).tolist()) != {ROLE}:
        raise ValueError("phase cache must contain only TRAIN rows")
    event_model = event_model.to(device).eval().requires_grad_(False)
    rgb_model = rgb_model.to(device).eval().requires_grad_(False)
    blocks: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            part = slice(start, min(start + batch_size, rows))
            eb, _, _ = frozen_context_state(
                event_model,
                *_tensor_batch(
                    event, ("features", "timing", "valid", "expert_phase"), part, device
                ),
            )
            rb, width, rout = frozen_context_state(
                rgb_model,
                *_tensor_batch(rgb, ("features", "timing", "valid", "expert_phase"), part, device),
            )
            features = torch.cat((rb, eb), dim=-1)
            zero_location = features[:, 160]
            zero_widths = 0.03 * functional.softplus(width)
            if not (
                torch.equal(zero_location, rout["raw_location"])
                and torch.equal(emitted_phase(zero_location), rout["point_phase"])
                and torch.equal(
                    (zero_location - zero_widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE), rout["q10"]
                )
                and torch.equal(
                    (zero_location + zero_widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE), rout["q90"]
                )
            ):
                raise AssertionError("zero-correction PHASE_ONLY does not reproduce frozen R_CTX")
            blocks.append(features.cpu().numpy().astype(np.float32, copy=False))
            widths.append(width.cpu().numpy().astype(np.float32, copy=False))
    values = {
        "features": np.concatenate(blocks),
        "rgb_width_logits": np.concatenate(widths),
        "target_phase": np.asarray(event["target_phase"], dtype=np.float32),
        "group_id": np.asarray(event["group_id"]).astype(str),
        "sample_token": np.asarray(event["sample_token"]).astype(str),
        "role": np.asarray(event["role"]).astype(str),
    }
    return _seal_cache(
        output,
        values,
        schema="final_refit_phase_cache_v1",
        parents=parents,
        normalizers=normalizers,
        compressed=compressed,
        native_train40_rows_sha256=native_train40_rows_sha256,
        details={
            "feature_schema_sha256": PHASE340_SCHEMA_SHA256,
            "feature_order": "R170_then_E170",
            "phase_only_selected_columns": [[160, 170], [330, 340]],
            "zero_correction_rgb_parity": True,
            "mixed_ancestry_explicit": True,
            "feature_width": PHASE_WIDTH,
        },
    )


__all__ = [
    "BLOCK_WIDTH",
    "EVENT_PHASE17_SCHEMA_SHA256",
    "PAIR_RGB_SCHEMA_SHA256",
    "PAIR_EVENT_SCHEMA_SHA256",
    "PHASE340_SCHEMA_SHA256",
    "RGB_PHASE17_SCHEMA_SHA256",
    "ROLE",
    "bind_files",
    "build_context_cache",
    "build_observation_cache",
    "build_pair_cache",
    "build_pair_cache_from_features",
    "build_phase_cache",
    "fit_train40_normalizer",
    "frozen_context_state",
    "load_sealed_cache",
    "normalizer_population_binding",
    "observation_identity_sha256",
    "ordered_row_identity_sha256",
    "sha256_file",
    "validate_cache_receipt",
]
