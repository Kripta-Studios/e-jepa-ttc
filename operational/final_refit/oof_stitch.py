"""Stitch only fold-invariant physical OOF interfaces in native TRAIN40 order.

PAIR133 tokens and context hidden160 states are intentionally excluded because
their coordinate systems are learned independently in each rotation.  The only
admitted cross-rotation payloads are raw PHASE17 physical observations and the
twenty physical phase/time coordinates consumed by PHASE_ONLY, plus its two RGB
width logits.  The latter requires a future explicit 20-coordinate head adapter;
this module does not fabricate a 340-D state or claim that such a head was fit.
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

from .oof_lineage import file_sha256, verify_endpoint_lineage, verify_stitch_receipt
from .recipe import canonical_sha256, ordered_row_identity_sha256, sha256

PHASE17_RAW_EVENT = "PHASE17_RAW_EVENT"
PHASE17_RAW_RGB = "PHASE17_RAW_RGB"
PHASE20_CONTEXT = "PHASE20_CONTEXT"
SEMANTICS = "physical_phase_clock_coordinates"
STITCH_SCHEMA = "final_refit_oof_stitch_v1"
FRAGMENT_SCHEMA = "final_refit_oof_fragment_v1"
NORMALIZER_SCHEMA = "final_refit_oof_normalizer_v1"
_WIDTHS = {PHASE17_RAW_EVENT: 17, PHASE17_RAW_RGB: 17, PHASE20_CONTEXT: 20}
_ENDPOINTS = {
    PHASE17_RAW_EVENT: ("E_A5", "E_C2F", "PAIR_E"),
    PHASE17_RAW_RGB: ("R_A5", "R_C2F", "PAIR_R"),
    PHASE20_CONTEXT: ("E_CTX", "R_CTX"),
}
_FORBIDDEN_WORDS = ("latent", "embedding", "hidden", "pair133")


def _strings(value: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    result = np.asarray(value).astype(str)
    if result.shape != (rows,) or any(not item for item in result.tolist()):
        raise ValueError(f"{name} must contain {rows} nonempty strings")
    return result


def _digest(value: object, *, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def payload_sha256(values: Mapping[str, np.ndarray]) -> str:
    """Hash names, dtypes, shapes and exact C-order bytes of a physical fragment."""

    digest = hashlib.sha256(b"final_refit_oof_physical_payload_v1\0")
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        for part in (name, array.dtype.str, json.dumps(array.shape, separators=(",", ":"))):
            encoded = part.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _expected_keys(interface: str) -> set[str]:
    common = {"sample_token", "group_id", "source_ordinal", "features"}
    if interface in {PHASE17_RAW_EVENT, PHASE17_RAW_RGB}:
        return common | {"timing", "valid", "expert_phase", "observation_id"}
    if interface == PHASE20_CONTEXT:
        return common | {"rgb_width_logits"}
    raise ValueError("OOF interface must be physical PHASE17 event/RGB or PHASE20 context")


def _validate_fragment(
    values: Mapping[str, np.ndarray], receipt: Mapping[str, Any], *, interface: str
) -> dict[str, np.ndarray]:
    expected_keys = _expected_keys(interface)
    if set(values) != expected_keys:
        difference = sorted(set(values) ^ expected_keys)
        raise ValueError(f"OOF physical fragment fields differ: {difference}")
    semantic_labels = (
        f"{receipt.get('interface', '')} {receipt.get('feature_semantics', '')}".lower()
    )
    if any(word in semantic_labels for word in _FORBIDDEN_WORDS):
        raise ValueError("OOF stitching forbids learned latent/embedding coordinates")
    rows = len(values["sample_token"])
    tokens = _strings(values["sample_token"], name="sample_token", rows=rows)
    groups = _strings(values["group_id"], name="group_id", rows=rows)
    ordinals = np.asarray(values["source_ordinal"], dtype=np.int64)
    features = np.asarray(values["features"], dtype=np.float32)
    if (
        rows <= 0
        or len(set(tokens.tolist())) != rows
        or ordinals.shape != (rows,)
        or len(set(ordinals.tolist())) != rows
        or np.any(ordinals < 0)
    ):
        raise ValueError("OOF physical fragment population or feature shape differs")
    normalized: dict[str, np.ndarray] = {
        "sample_token": tokens,
        "group_id": groups,
        "source_ordinal": ordinals,
    }
    if interface in {PHASE17_RAW_EVENT, PHASE17_RAW_RGB}:
        valid = np.asarray(values["valid"], dtype=np.bool_)
        timing = np.asarray(values["timing"], dtype=np.float32)
        observation_id = np.asarray(values["observation_id"]).astype(str)
        expert = np.asarray(values["expert_phase"], dtype=np.float32)
        if (
            features.ndim != 3
            or features.shape[0] != rows
            or features.shape[2] != 17
            or features.shape[1] not in range(1, 9)
            or valid.shape != features.shape[:2]
            or timing.shape != (*features.shape[:2], 4)
            or observation_id.shape != features.shape[:2]
            or expert.shape != (rows, 3)
            or not np.all(valid[:, -1])
            or np.any(valid[:, :-1] & ~valid[:, 1:])
            or any(not item for item in observation_id[valid].tolist())
            or not np.isfinite(features[valid]).all()
            or not np.isfinite(timing[valid]).all()
            or not np.isfinite(expert).all()
        ):
            raise ValueError("PHASE17_RAW requires causal histories, clocks and expert phases")
        normalized.update(
            {
                "features": np.ascontiguousarray(features),
                "timing": np.ascontiguousarray(timing),
                "valid": np.ascontiguousarray(valid),
                "expert_phase": np.ascontiguousarray(expert),
                "observation_id": np.ascontiguousarray(observation_id),
            }
        )
    else:
        if features.shape != (rows, 20) or not np.isfinite(features).all():
            raise ValueError("PHASE20_CONTEXT requires finite physical inputs [N,20]")
        normalized["features"] = np.ascontiguousarray(features)
        widths = np.asarray(values["rgb_width_logits"], dtype=np.float32)
        if widths.shape != (rows, 2) or not np.isfinite(widths).all():
            raise ValueError("PHASE20_CONTEXT requires finite RGB width logits [N,2]")
        normalized["rgb_width_logits"] = np.ascontiguousarray(widths)

    required = {
        "schema",
        "status",
        "interface",
        "feature_semantics",
        "width",
        "rotation",
        "role",
        "generation_manifest_sha256",
        "ordered_sample_token_sha256",
        "group_ids",
        "payload_sha256",
        "endpoint_lineages",
    }
    if not required.issubset(receipt):
        raise ValueError(f"OOF fragment receipt lacks fields: {sorted(required - set(receipt))}")
    generation_groups = sorted(set(groups.tolist()))
    if (
        receipt["schema"] != FRAGMENT_SCHEMA
        or receipt["status"] != "COMPLETE"
        or receipt["interface"] != interface
        or receipt["feature_semantics"] != SEMANTICS
        or receipt["width"] != _WIDTHS[interface]
        or receipt["role"] != "generation"
        or int(receipt["rotation"]) not in range(5)
        or sorted(str(item) for item in receipt["group_ids"]) != generation_groups
        or receipt["ordered_sample_token_sha256"] != canonical_sha256(tokens.tolist())
        or receipt["payload_sha256"] != payload_sha256(normalized)
    ):
        raise ValueError("OOF fragment identity, role, ancestry or physical interface differs")
    _digest(receipt["generation_manifest_sha256"], name="generation manifest")
    references = receipt["endpoint_lineages"]
    if not isinstance(references, dict) or set(references) != set(_ENDPOINTS[interface]):
        raise ValueError("OOF fragment endpoint lineages differ from its physical interface")
    for stage_id, reference in references.items():
        if not isinstance(reference, dict):
            raise ValueError("OOF endpoint lineage reference must be a mapping")
        path = Path(str(reference.get("path"))).resolve()
        if file_sha256(path) != reference.get("file_sha256"):
            raise ValueError("OOF endpoint lineage file changed")
        endpoint = verify_endpoint_lineage(path)
        ancestry = endpoint.get("training_feature_ancestry")
        if (
            endpoint.get("stage_id") != stage_id
            or endpoint.get("rotation") != int(receipt["rotation"])
            or endpoint.get("lineage_sha256") != reference.get("lineage_sha256")
            or not isinstance(ancestry, dict)
            or ancestry.get("generation_rows_excluded_from_all_ancestors") is not True
            or endpoint.get("deployment_dependencies") != {}
        ):
            raise ValueError("OOF endpoint lineage does not prove generation exclusion")
    return normalized


def stitch_oof_physical(
    native_tokens: Sequence[str],
    native_groups: Sequence[str],
    fragments: Sequence[tuple[Mapping[str, np.ndarray], Mapping[str, Any]]],
    *,
    interface: str,
    fold_contract_sha256: str,
    expected_population: int = 88_744,
    expected_groups: int = 40,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate five generation folds and reorder their physical rows natively."""

    width = _WIDTHS.get(interface)
    if width is None or len(fragments) != 5:
        raise ValueError("OOF stitch requires one physical interface and exactly five rotations")
    _digest(fold_contract_sha256, name="fold contract")
    tokens = [str(item) for item in native_tokens]
    groups = [str(item) for item in native_groups]
    if (
        len(tokens) != expected_population
        or len(groups) != expected_population
        or len(set(tokens)) != expected_population
        or len(set(groups)) != expected_groups
    ):
        raise ValueError("native OOF inventory differs from the required population")

    admitted: list[dict[str, np.ndarray]] = []
    fragment_receipts: list[dict[str, Any]] = []
    rotations: set[int] = set()
    for values, receipt in fragments:
        normalized = _validate_fragment(values, receipt, interface=interface)
        rotation = int(receipt["rotation"])
        if rotation in rotations:
            raise ValueError("OOF stitch contains a duplicate rotation")
        rotations.add(rotation)
        admitted.append(normalized)
        fragment_receipts.append(
            {
                "rotation": rotation,
                "generation_manifest_sha256": receipt["generation_manifest_sha256"],
                "ordered_sample_token_sha256": receipt["ordered_sample_token_sha256"],
                "group_ids": sorted(str(item) for item in receipt["group_ids"]),
                "payload_sha256": receipt["payload_sha256"],
                "endpoint_lineages": receipt["endpoint_lineages"],
            }
        )
    if rotations != set(range(5)):
        raise ValueError("OOF stitch must contain rotations 0 through 4 exactly once")

    merged = {
        name: np.concatenate([fragment[name] for fragment in admitted])
        for name in _expected_keys(interface)
    }
    merged_tokens = merged["sample_token"].astype(str).tolist()
    ordinals = np.asarray(merged["source_ordinal"], dtype=np.int64)
    if (
        len(merged_tokens) != expected_population
        or len(set(merged_tokens)) != expected_population
        or set(ordinals.tolist()) != set(range(expected_population))
    ):
        raise ValueError("OOF generation fragments do not cover every native row exactly once")
    order = np.argsort(ordinals, kind="stable")
    stitched = {name: np.ascontiguousarray(value[order]) for name, value in merged.items()}
    if (
        stitched["source_ordinal"].tolist() != list(range(expected_population))
        or stitched["sample_token"].astype(str).tolist() != tokens
        or stitched["group_id"].astype(str).tolist() != groups
    ):
        raise ValueError("OOF stitch token/group rows differ from native TRAIN40 order")
    stitch_receipt: dict[str, Any] = {
        "schema": STITCH_SCHEMA,
        "status": "STRUCTURAL_PHYSICAL_STITCH_NOT_ADMITTED",
        "interface": interface,
        "feature_semantics": SEMANTICS,
        "width": width,
        "population": expected_population,
        "groups": expected_groups,
        "fold_contract_sha256": fold_contract_sha256,
        "ordered_row_identity_sha256": ordered_row_identity_sha256(tokens, groups),
        "token_set_sha256": canonical_sha256(sorted(tokens)),
        "fragments": sorted(fragment_receipts, key=lambda item: int(item["rotation"])),
        "payload_sha256": payload_sha256(stitched),
        "cross_fold_learned_coordinates": False,
        "head_adapter_required": interface == PHASE20_CONTEXT,
        "head_adapter_status": (
            "REQUIRED_NOT_IMPLEMENTED" if interface == PHASE20_CONTEXT else "NOT_APPLICABLE"
        ),
        "scientific_launch_authorized": False,
        "blind_validation_claim": False,
        "integration_scope": "STRUCTURAL_ARRAY_CONTRACT_ONLY",
    }
    return stitched, stitch_receipt


def stitch_from_verified_lineage(
    lineage_receipt: Mapping[str, Any] | Path,
    native_tokens: Sequence[str],
    native_groups: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load fragment bytes only after the independent OOF lineage authority passes."""

    authority = verify_stitch_receipt(lineage_receipt)
    if (
        authority.get("status")
        != "VERIFIED_STRUCTURAL_OOF_STITCH_NOT_SCIENTIFICALLY_ADMITTED"
        or authority.get("scientific_launch_authorized") is not False
        or authority.get("blind_validation_claim") is not False
        or authority.get("targets_read_for_stitching") is not False
    ):
        raise ValueError("OOF lineage authority status or scientific scope differs")
    interface = str(authority["interface"])
    fragments: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []
    for fragment in authority["fragments"]:
        payload_path = Path(str(fragment["payload_path"])).resolve(strict=True)
        with np.load(payload_path, allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        synthetic_receipt = {
            "schema": FRAGMENT_SCHEMA,
            "status": "COMPLETE",
            "interface": interface,
            "feature_semantics": SEMANTICS,
            "width": _WIDTHS[interface],
            "rotation": int(fragment["rotation"]),
            "role": "generation",
            "generation_manifest_sha256": fragment["generation_rows_sha256"],
            "ordered_sample_token_sha256": fragment["ordered_sample_token_sha256"],
            "group_ids": fragment["group_ids"],
            "payload_sha256": payload_sha256(values),
            "endpoint_lineages": fragment["endpoint_lineages"],
        }
        fragments.append((values, synthetic_receipt))
    stitched, receipt = stitch_oof_physical(
        native_tokens,
        native_groups,
        fragments,
        interface=interface,
        fold_contract_sha256=str(authority["fold_contract_sha256"]),
        expected_population=int(authority["population"]),
        expected_groups=int(authority["groups"]),
    )
    receipt["lineage_authority_sha256"] = str(authority["stitch_sha256"])
    receipt["lineage_authority_verified"] = True
    receipt["lineage_authority_status"] = str(authority["status"])
    receipt["status"] = "LINEAGE_VERIFIED_NOT_SCIENTIFICALLY_ADMITTED"
    receipt["integration_scope"] = (
        "LINEAGE_BOUND_TRAINING_FEATURES_NOT_SCIENTIFICALLY_ADMITTED"
    )
    return stitched, receipt


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_oof_stitch(
    output: Path, values: Mapping[str, np.ndarray], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal a validated stitch; this performs no inference or optimizer update."""

    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace OOF stitch {output}")
    if receipt.get("payload_sha256") != payload_sha256(values):
        raise ValueError("OOF stitch values differ from their receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    with temporary.open("xb") as stream:
        cast(Any, np.savez_compressed)(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    sealed = {**receipt, "output": str(output.resolve()), "output_sha256": sha256(output)}
    _atomic_json(output.with_suffix(".json"), sealed)
    return sealed


def fit_oof_phase17_normalizer(stitch_path: Path, output: Path) -> dict[str, Any]:
    """Fit a fresh global normalizer only from a sealed PHASE17_RAW OOF stitch."""

    stitch = stitch_path.resolve(strict=True)
    receipt_path = stitch.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != STITCH_SCHEMA
        or receipt.get("status") != "LINEAGE_VERIFIED_NOT_SCIENTIFICALLY_ADMITTED"
        or receipt.get("interface") not in {PHASE17_RAW_EVENT, PHASE17_RAW_RGB}
        or receipt.get("feature_semantics") != SEMANTICS
        or receipt.get("width") != 17
        or receipt.get("output_sha256") != sha256(stitch)
        or receipt.get("lineage_authority_verified") is not True
        or receipt.get("lineage_authority_status")
        != "VERIFIED_STRUCTURAL_OOF_STITCH_NOT_SCIENTIFICALLY_ADMITTED"
    ):
        raise ValueError("PHASE17 OOF stitch lacks verified structural lineage authority")
    _digest(receipt.get("lineage_authority_sha256"), name="lineage authority")
    with np.load(stitch, allow_pickle=False) as archive:
        if set(archive.files) != _expected_keys(str(receipt["interface"])):
            raise ValueError("PHASE17 OOF stitch fields differ")
        values = {name: archive[name] for name in archive.files}
    if receipt.get("payload_sha256") != payload_sha256(values):
        raise ValueError("PHASE17 OOF stitch payload changed")
    histories = np.asarray(values["features"], dtype=np.float64)
    valid = np.asarray(values["valid"], dtype=np.bool_)
    observation_ids = np.asarray(values["observation_id"]).astype(str)
    groups = np.asarray(values["group_id"]).astype(str)
    unique: dict[str, tuple[str, np.ndarray]] = {}
    for row, column in zip(*np.nonzero(valid), strict=True):
        identity = str(observation_ids[row, column])
        group = str(groups[row])
        feature = np.asarray(histories[row, column])
        previous = unique.setdefault(identity, (group, feature))
        if previous[0] != group or not np.array_equal(previous[1], feature):
            raise ValueError("one OOF observation has conflicting group or PHASE17 values")
    if not unique:
        raise ValueError("PHASE17 OOF stitch contains no valid observations")
    features = np.stack([unique[identity][1] for identity in sorted(unique)])
    mean = features.mean(axis=0)
    standard_deviation = features.std(axis=0)
    scale = np.where(standard_deviation > 1e-6, standard_deviation, 1.0)
    payload: dict[str, Any] = {
        "schema": NORMALIZER_SCHEMA,
        "status": "STRUCTURAL_NORMALIZER_NOT_SCIENTIFICALLY_ADMITTED",
        "interface": receipt["interface"],
        "feature_semantics": SEMANTICS,
        "stitch_sha256": sha256(stitch),
        "fold_contract_sha256": receipt["fold_contract_sha256"],
        "targets_read": False,
        "scientific_launch_authorized": False,
        "blind_validation_claim": False,
        "integration_scope": "OOF_NORMALIZER_NOT_SCIENTIFICALLY_ADMITTED",
        "population": receipt["population"],
        "groups": receipt["groups"],
        "token_set_sha256": receipt["token_set_sha256"],
        "feature_width": 17,
        "unique_observations": len(unique),
        "observation_set_sha256": canonical_sha256(sorted(unique)),
        "parents": {
            "oof_stitch": sha256(stitch),
            "oof_stitch_receipt": sha256(receipt_path),
        },
        "mean": mean.astype(np.float32).tolist(),
        "scale": scale.astype(np.float32).tolist(),
    }
    payload["normalizer_payload_sha256"] = canonical_sha256(payload)
    if output.exists():
        raise FileExistsError(f"refusing to replace OOF normalizer {output}")
    _atomic_json(output, payload)
    return payload


__all__ = [
    "FRAGMENT_SCHEMA",
    "NORMALIZER_SCHEMA",
    "PHASE17_RAW_EVENT",
    "PHASE17_RAW_RGB",
    "PHASE20_CONTEXT",
    "SEMANTICS",
    "STITCH_SCHEMA",
    "fit_oof_phase17_normalizer",
    "payload_sha256",
    "stitch_oof_physical",
    "stitch_from_verified_lineage",
    "write_oof_stitch",
]
