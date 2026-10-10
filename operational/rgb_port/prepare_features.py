"""Seal RGB-PORT PHASE17 observation, history, and fusion caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from e_jepa_ttc.rgb_port.features import (
    EVENT_PHASE17_SHA256,
    PAIR_E_SCHEMA_SHA256,
    PAIR_R_SCHEMA_SHA256,
    RGB_PHASE17_SHA256,
    event_phase17,
    pair133,
)
from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer, fit_normalizer

ENDPOINT_IDS = {
    "E_A5_MATCHED",
    "E_C2F_MATCHED",
    "R_A5",
    "R_C2F",
    "PAIR_E_MATCHED",
    "PAIR_R",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    with temporary.open("xb") as stream:
        savez = cast(Any, np.savez)
        savez(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_endpoint_freeze(path: Path) -> dict[str, str]:
    """Require all own producers and PAIR endpoints before any H/V feature exists."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, dict) or set(endpoints) != ENDPOINT_IDS:
        raise ValueError("feature freeze must bind all four producers and both PAIR endpoints")
    result: dict[str, str] = {}
    for fit_id, item in endpoints.items():
        if item.get("status") != "COMPLETE" or not item.get("scientific_endpoint"):
            raise ValueError(f"{fit_id} is not a frozen endpoint")
        checkpoint = Path(item["checkpoint_path"]).resolve(strict=True)
        if item.get("checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError(f"{fit_id} checkpoint changed")
        result[fit_id] = item["checkpoint_sha256"]
    return result


def build_pair_cache(
    source_path: Path, output: Path, *, modality: str, producer_receipt: Path
) -> dict[str, Any]:
    """Build the P-only 133-D cache from the frozen modality-local A5 output."""
    if modality not in {"event", "rgb"}:
        raise ValueError("PAIR modality must be event or rgb")
    receipt = json.loads(producer_receipt.read_text(encoding="utf-8"))
    expected = "E_A5_MATCHED" if modality == "event" else "R_A5"
    if (
        receipt.get("fit_id") != expected
        or receipt.get("status") != "COMPLETE"
        or not receipt.get("scientific_endpoint")
    ):
        raise ValueError("PAIR cache requires its completed P-trained A5 endpoint")
    checkpoint = Path(receipt["checkpoint_path"]).resolve(strict=True)
    if receipt.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("PAIR parent checkpoint changed")
    with np.load(source_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    required = {
        "token128",
        "delta_t_s",
        "support",
        "target_phase",
        "mass",
        "sequence_id",
        "sample_token",
        "group_id",
        "role",
    }
    if set(values) != required or set(np.asarray(values["role"]).astype(str)) != {"P"}:
        raise ValueError("PAIR source schema/role differs")
    token = torch.from_numpy(np.asarray(values["token128"], np.float32))
    delta_values = np.asarray(values["delta_t_s"], np.float32)
    if delta_values.ndim == 2 and delta_values.shape[1] in {1, 2}:
        delta_values = delta_values[:, -1]
    if delta_values.shape != (len(token),):
        raise ValueError("PAIR delta must preserve the current previous-to-current interval")
    delta = torch.from_numpy(delta_values)
    support = torch.from_numpy(np.asarray(values["support"], np.float32))
    if support.ndim != 2 or support.shape[1] not in {2, 3}:
        raise ValueError("PAIR support must preserve T2/T3")
    current = support[:, -1]
    supports = torch.stack((current, torch.minimum(support[:, -2], current)), -1)
    result = {
        "features": pair133(token, delta, supports).numpy(),
        "target_phase": np.asarray(values["target_phase"], np.float32),
        "mass": np.asarray(values["mass"], np.float32),
        "sequence_id": values["sequence_id"],
        "sample_token": values["sample_token"],
        "group_id": values["group_id"],
        "role": values["role"],
    }
    _atomic_npz(output, result)
    schema = PAIR_E_SCHEMA_SHA256 if modality == "event" else PAIR_R_SCHEMA_SHA256
    result_receipt = {
        "schema": "rgb_port_pair_cache_v1",
        "status": "COMPLETE",
        "role": "P",
        "modality": modality,
        "rows": len(token),
        "schema_sha256": schema,
        "parent_checkpoint_sha256": receipt["checkpoint_sha256"],
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(result_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result_receipt


def fit_h_normalizer(observations_path: Path, output: Path, *, modality: str) -> dict[str, Any]:
    """Fit mean/scale once per unique H observation, never per repeated query row."""
    with np.load(observations_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    if set(np.asarray(values["role"]).astype(str)) != {"H"}:
        raise ValueError("normalization is H-only")
    identities = np.asarray(values["observation_id"]).astype(str)
    if len(set(identities.tolist())) != len(identities):
        raise ValueError("normalization input must contain unique observations")
    features = np.asarray(values["features"], np.float64)
    if features.shape != (len(identities), 17) or not np.isfinite(features).all():
        raise ValueError("normalization feature matrix differs")
    schema = EVENT_PHASE17_SHA256 if modality == "event" else RGB_PHASE17_SHA256
    normalizer = fit_normalizer(
        features,
        identities.tolist(),
        fit_role="H",
        expected_role="H",
        producer_sha256=str(values["endpoint_freeze_sha256"]),
        schema_sha256=schema,
        modality=modality,
    )
    temporary = output.with_suffix(f".{uuid.uuid4().hex}.tmp")
    normalizer.save(temporary)
    os.replace(temporary, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {**payload, "output": str(output.resolve()), "output_sha256": _sha256(output)}


def build_observations(
    blocks_path: Path, output: Path, *, role: str, modality: str, freeze: Path
) -> dict[str, Any]:
    """Build raw PHASE17 exactly once per distinct frozen observation."""
    if role not in {"H", "V"} or modality not in {"event", "rgb"}:
        raise ValueError("only H/V and event/rgb are valid")
    endpoint_hashes = validate_endpoint_freeze(freeze)
    with np.load(blocks_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    required = {
        "observation_id",
        "base_statistics2",
        "a5_diagnostics3",
        "c2f_diagnostics3",
        "expert_phase",
        "anchor_us",
        "available_us",
        "role",
    }
    if set(values) != required:
        raise ValueError(f"observation block schema differs: {sorted(set(values) ^ required)}")
    identities = np.asarray(values["observation_id"]).astype(str)
    if len(set(identities.tolist())) != len(identities):
        raise ValueError("observation feature generation requires unique identities")
    if set(np.asarray(values["role"]).astype(str).tolist()) != {role}:
        raise ValueError("observation roles differ")
    base = torch.from_numpy(np.asarray(values["base_statistics2"], dtype=np.float32))
    a5 = torch.from_numpy(np.asarray(values["a5_diagnostics3"], dtype=np.float32))
    c2f = torch.from_numpy(np.asarray(values["c2f_diagnostics3"], dtype=np.float32))
    expert = torch.from_numpy(np.asarray(values["expert_phase"], dtype=np.float32))
    if modality == "event":
        features = event_phase17(base, a5, c2f, expert)
        schema = EVENT_PHASE17_SHA256
    else:
        # RGB base statistics were computed from raw [0,1] pixels by the producer pass.
        phase = expert
        signed = torch.stack(
            (phase[:, 2] - phase[:, 0], phase[:, 2] - phase[:, 1], phase[:, 1] - phase[:, 0]), -1
        )
        features = torch.cat((base, a5, c2f, phase, signed, signed.abs()), -1)
        schema = RGB_PHASE17_SHA256
    output_values = {
        "observation_id": identities,
        "features": features.numpy().astype(np.float32),
        "anchor_us": np.asarray(values["anchor_us"], dtype=np.int64),
        "available_us": np.asarray(values["available_us"], dtype=np.int64),
        "role": np.full(len(identities), role),
        "schema_sha256": np.asarray(schema),
        "endpoint_freeze_sha256": np.asarray(_sha256(freeze)),
    }
    if np.any(output_values["available_us"] < output_values["anchor_us"]):
        raise ValueError("observation availability precedes anchor")
    _atomic_npz(output, output_values)
    receipt = {
        "schema": "rgb_port_observation_features_v1",
        "status": "COMPLETE",
        "role": role,
        "modality": modality,
        "rows": len(identities),
        "schema_sha256": schema,
        "endpoint_hashes": endpoint_hashes,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _normalizer(
    path: Path, *, schema: str, modality: str, producer_sha256: str
) -> tuple[np.ndarray, np.ndarray, str]:
    value = FrozenNormalizer.load(path)
    value.validate_endpoint(
        modality=modality,
        fit_role="H",
        schema_sha256=schema,
        producer_sha256=producer_sha256,
    )
    if value.mean.shape != (17,):
        raise ValueError("normalizer statistics differ")
    return value.mean, value.scale, _sha256(path)


def build_history(
    observations_path: Path,
    history_path: Path,
    normalizer_path: Path,
    output: Path,
    *,
    role: str,
    modality: str,
    freeze: Path,
) -> dict[str, Any]:
    """Gather causal histories by observation identity and apply the frozen H transform."""
    validate_endpoint_freeze(freeze)
    with np.load(observations_path, allow_pickle=False) as archive:
        observations = {name: archive[name] for name in archive.files}
    with np.load(history_path, allow_pickle=False) as archive:
        history = {name: archive[name] for name in archive.files}
    required = {
        "query_id",
        "observation_id",
        "timing",
        "valid",
        "current_expert_phase",
        "target_phase",
        "mass",
        "sample_token",
        "group_id",
        "role",
    }
    if set(history) != required:
        raise ValueError(f"history schema differs: {sorted(set(history) ^ required)}")
    schema = EVENT_PHASE17_SHA256 if modality == "event" else RGB_PHASE17_SHA256
    if str(observations["schema_sha256"]) != schema:
        raise ValueError("observation cache schema differs from requested modality")
    if str(observations["endpoint_freeze_sha256"]) != _sha256(freeze):
        raise ValueError("observation cache endpoint ancestry differs")
    if set(np.asarray(observations["role"]).astype(str)) != {role} or set(
        np.asarray(history["role"]).astype(str)
    ) != {role}:
        raise ValueError("history/observation role differs")
    mean, scale, normalizer_sha = _normalizer(
        normalizer_path,
        schema=schema,
        modality=modality,
        producer_sha256=str(observations["endpoint_freeze_sha256"]),
    )
    ids = np.asarray(observations["observation_id"]).astype(str)
    lookup = {identity: index for index, identity in enumerate(ids)}
    requested = np.asarray(history["observation_id"]).astype(str)
    valid = np.asarray(history["valid"], dtype=np.bool_)
    if (
        requested.shape != valid.shape
        or not np.all(valid[:, -1])
        or np.any(valid[:, :-1] & ~valid[:, 1:])
    ):
        raise ValueError("history observation IDs/masks differ")
    features = np.zeros((*requested.shape, 17), dtype=np.float32)
    for row, column in zip(*np.nonzero(valid), strict=True):
        identity = requested[row, column]
        if identity not in lookup:
            raise ValueError(f"history references unknown observation {identity}")
        features[row, column] = observations["features"][lookup[identity]]
    features[valid] = (features[valid] - mean) / scale
    expert_phase = np.asarray(history["current_expert_phase"], np.float32)
    result = {
        "features": features,
        "timing": np.asarray(history["timing"], np.float32),
        "valid": valid,
        "expert_phase": expert_phase.astype(np.float32),
        "target_phase": np.asarray(history["target_phase"], np.float32),
        "mass": np.asarray(history["mass"], np.float32),
        "sample_token": history["sample_token"],
        "group_id": history["group_id"],
        "role": np.full(len(features), role),
    }
    _atomic_npz(output, result)
    receipt = {
        "schema": "rgb_port_history_features_v1",
        "status": "COMPLETE",
        "role": role,
        "modality": modality,
        "queries": len(features),
        "normalizer_sha256": normalizer_sha,
        "source_sha256": _sha256(observations_path),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def build_fusion(event_path: Path, rgb_path: Path, output: Path) -> dict[str, Any]:
    """Join independently normalized E/R caches without changing either clock."""
    with np.load(event_path, allow_pickle=False) as archive:
        event = {name: archive[name] for name in archive.files}
    with np.load(rgb_path, allow_pickle=False) as archive:
        rgb = {name: archive[name] for name in archive.files}
    if not np.array_equal(event["sample_token"], rgb["sample_token"]):
        raise ValueError("event/RGB history caches are not query aligned")
    for name in ("target_phase", "mass", "group_id", "role"):
        if not np.array_equal(event[name], rgb[name]):
            raise ValueError(f"fusion supervision differs: {name}")
    result = {
        f"event_{name}": event[name] for name in ("features", "timing", "valid", "expert_phase")
    }
    result.update({f"rgb_{name}": rgb[name] for name in ("features", "timing", "valid")})
    result.update(
        {name: event[name] for name in ("target_phase", "mass", "sample_token", "group_id", "role")}
    )
    _atomic_npz(output, result)
    receipt = {
        "schema": "rgb_port_fusion_features_v1",
        "status": "COMPLETE",
        "queries": len(event["target_phase"]),
        "event_sha256": _sha256(event_path),
        "rgb_sha256": _sha256(rgb_path),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--source", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--modality", choices=("event", "rgb"), required=True)
    pair.add_argument("--producer-receipt", type=Path, required=True)
    obs = sub.add_parser("observations")
    obs.add_argument("--blocks", type=Path, required=True)
    obs.add_argument("--output", type=Path, required=True)
    obs.add_argument("--role", choices=("H", "V"), required=True)
    obs.add_argument("--modality", choices=("event", "rgb"), required=True)
    obs.add_argument("--endpoint-freeze", type=Path, required=True)
    hist = sub.add_parser("history")
    hist.add_argument("--observations", type=Path, required=True)
    hist.add_argument("--history", type=Path, required=True)
    hist.add_argument("--normalizer", type=Path, required=True)
    hist.add_argument("--output", type=Path, required=True)
    hist.add_argument("--role", choices=("H", "V"), required=True)
    hist.add_argument("--modality", choices=("event", "rgb"), required=True)
    hist.add_argument("--endpoint-freeze", type=Path, required=True)
    norm = sub.add_parser("normalizer")
    norm.add_argument("--observations", type=Path, required=True)
    norm.add_argument("--output", type=Path, required=True)
    norm.add_argument("--modality", choices=("event", "rgb"), required=True)
    fusion = sub.add_parser("fusion")
    fusion.add_argument("--event-cache", type=Path, required=True)
    fusion.add_argument("--rgb-cache", type=Path, required=True)
    fusion.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "pair":
        result = build_pair_cache(
            args.source,
            args.output,
            modality=args.modality,
            producer_receipt=args.producer_receipt,
        )
    elif args.command == "observations":
        result = build_observations(
            args.blocks,
            args.output,
            role=args.role,
            modality=args.modality,
            freeze=args.endpoint_freeze,
        )
    elif args.command == "history":
        result = build_history(
            args.observations,
            args.history,
            args.normalizer,
            args.output,
            role=args.role,
            modality=args.modality,
            freeze=args.endpoint_freeze,
        )
    elif args.command == "normalizer":
        result = fit_h_normalizer(args.observations, args.output, modality=args.modality)
    else:
        result = build_fusion(args.event_cache, args.rgb_cache, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_fusion",
    "build_history",
    "build_observations",
    "build_pair_cache",
    "fit_h_normalizer",
    "main",
    "validate_endpoint_freeze",
]
