"""Durable stage-by-stage materialization for the final TRAIN40 feature chain."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.rgb_port.data import TarFrameReader, decode_query
from e_jepa_ttc.rgb_port.features import pair133, raw_rgb_statistics
from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.rgb_port.accounting import atomic_write_json
from operational.rgb_port.infer_experts import clock_features, delta_matrix
from operational.rgb_port.train_heads import pair_point_phase
from operational.rgb_port.train_producers import producer_features

from .features import (
    ROLE,
    bind_files,
    build_context_cache,
    build_observation_cache,
    build_pair_cache_from_features,
    build_phase_cache,
    fit_train40_normalizer,
    normalizer_population_binding,
    sha256_file,
    validate_cache_receipt,
)
from .producers import build_rgb_recipe, load_rgb_endpoint
from .recipe import load_recipe, ordered_row_identity_sha256, stage
from .rgb import AllTrainRGBSource, load_all_train_source
from .training import load_cached_endpoint


class QueryProducerSource(Protocol):
    """Minimum source interface needed for label-free query feature extraction."""

    population_size: int

    @property
    def batch_keys(self) -> Sequence[tuple[str, int]]: ...

    def batch(self, indices: Sequence[int], modality: str) -> ObjectEventV4Batch: ...

    def row(self, index: int) -> Mapping[str, Any]: ...


def _canonical_sha(value: object) -> str:
    return (
        __import__("hashlib")
        .sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        .hexdigest()
    )


def _existing_cache(
    output: Path,
    schema: str,
    parents: Mapping[str, str],
    *,
    modality: str | None = None,
    native_train40_rows_sha256: str | None = None,
    expected_ordered_row_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Reuse a complete cache only for the exact requested producer lineage."""
    receipt = validate_cache_receipt(output, schema)
    if receipt["parents"] != dict(parents):
        raise ValueError("existing materialization parent bindings differ")
    if modality is not None and receipt.get("modality") != modality:
        raise ValueError("existing materialization modality differs")
    if native_train40_rows_sha256 is not None and (
        receipt.get("native_train40_rows_sha256") != native_train40_rows_sha256
        or receipt.get("ordered_row_identity_sha256") != expected_ordered_row_identity_sha256
    ):
        raise ValueError("existing materialization TRAIN40 population differs")
    return receipt


def _validate_raw_history_receipt(
    history_path: Path,
    *,
    modality: str,
    parents: Mapping[str, str],
    observation_receipt: Mapping[str, Any],
    history: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    receipt_path = history_path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = len(history["sample_token"])
    groups = sorted(set(np.asarray(history["group_id"]).astype(str).tolist()))
    required = {
        "schema",
        "status",
        "role",
        "modality",
        "queries",
        "observations",
        "groups",
        "parents",
        "observations_sha256",
        "history_sha256",
        "optimizer_updates",
    }
    if (
        not required.issubset(receipt)
        or receipt.get("schema") != "final_refit_raw_history_v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("role") != ROLE
        or receipt.get("modality") != modality
        or receipt.get("queries") != rows
        or receipt.get("observations") != observation_receipt["rows"]
        or receipt.get("groups") != len(groups)
        or receipt.get("parents") != dict(sorted(parents.items()))
        or receipt.get("observations_sha256") != observation_receipt["output_sha256"]
        or receipt.get("history_sha256") != sha256_file(history_path)
        or receipt.get("optimizer_updates") != 0
    ):
        raise ValueError("raw history receipt identity, ancestry or contents differ")
    return receipt


class FragmentStore:
    """Append-only inference shards whose binding is checked on every resume."""

    def __init__(self, root: Path, binding: Mapping[str, Any]) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = dict(binding)
        self.binding_sha256 = _canonical_sha(self.binding)
        path = self.root / "BINDING.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != self.binding:
                raise ValueError("materialization fragment binding changed")
        else:
            atomic_write_json(path, self.binding)

    def read(self, part: int) -> dict[str, np.ndarray] | None:
        path = self.root / f"part_{part:06d}.npz"
        receipt_path = path.with_suffix(".json")
        if not receipt_path.exists():
            return None
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("binding_sha256") != self.binding_sha256 or receipt.get(
            "sha256"
        ) != sha256_file(path):
            raise ValueError("materialization fragment changed after completion")
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def write(self, part: int, values: Mapping[str, np.ndarray]) -> None:
        path = self.root / f"part_{part:06d}.npz"
        if path.with_suffix(".json").exists():
            raise FileExistsError("refusing to replace a materialization fragment")
        if path.exists():
            # A crash can occur between publishing bytes and publishing the receipt.
            # Recompute the fragment and admit only exact equality; never overwrite it.
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(values) or any(
                    not np.array_equal(archive[name], values[name]) for name in values
                ):
                    raise ValueError("uncommitted fragment differs from recomputed values")
        else:
            temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
            cast(Callable[..., None], np.savez_compressed)(temporary, **values)
            os.replace(temporary, path)
        atomic_write_json(
            path.with_suffix(".json"),
            {
                "binding_sha256": self.binding_sha256,
                "sha256": sha256_file(path),
                "rows": len(next(iter(values.values()))),
                "optimizer_updates": 0,
            },
        )


def materialize_query_pair_features(
    source: QueryProducerSource,
    model: nn.Module,
    output: Path,
    *,
    modality: str,
    parents: Mapping[str, str],
    native_train40_rows_sha256: str,
    expected_ordered_row_identity_sha256: str,
    device: str = "cpu",
    batch_size: int = 32,
    rows_per_fragment: int = 256,
    guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute a frozen A5 over all TRAIN40 queries and seal the canonical PAIR cache."""

    if modality not in {"event", "rgb"} or batch_size <= 0 or rows_per_fragment <= 0:
        raise ValueError("invalid query feature materialization arguments")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA query materialization requested but unavailable")
    if output.exists():
        return _existing_cache(
            output,
            "final_refit_pair_cache_v1",
            parents,
            modality=modality,
            native_train40_rows_sha256=native_train40_rows_sha256,
            expected_ordered_row_identity_sha256=expected_ordered_row_identity_sha256,
        )
    parent_hashes = dict(sorted(parents.items()))
    store = FragmentStore(
        output.with_suffix(".parts"),
        {
            "schema": "final_refit_pair_fragments_v1",
            "modality": modality,
            "population": source.population_size,
            "parents": parent_hashes,
            "device": device,
            "precision": "float32",
            "targets_are_model_inputs": False,
        },
    )
    model = model.to(device).float().eval().requires_grad_(False)
    parts: list[dict[str, np.ndarray]] = []
    with torch.inference_mode():
        for part, start in enumerate(range(0, source.population_size, rows_per_fragment)):
            if guard is not None:
                guard()
            saved = store.read(part)
            if saved is not None:
                parts.append(saved)
                continue
            stop = min(start + rows_per_fragment, source.population_size)
            indices = list(range(start, stop))
            produced: dict[int, np.ndarray] = {}
            grouped: dict[tuple[str, int], list[int]] = {}
            for index in indices:
                grouped.setdefault(source.batch_keys[index], []).append(index)
            for selected in grouped.values():
                for offset in range(0, len(selected), batch_size):
                    if guard is not None:
                        guard()
                    query_ids = selected[offset : offset + batch_size]
                    batch = source.batch(query_ids, modality)
                    sensor = batch.events.to(device).float()
                    delta = delta_matrix(batch.delta_t_s, sensor.shape[1]).to(device)
                    result = producer_features(cast(CausalScaleTTC, model), sensor, delta)
                    current = result["support"][:, -1]
                    feature = torch.cat(
                        (
                            result["token128"],
                            delta[:, -1, None],
                            torch.log(delta[:, -1, None]),
                            delta[:, -1, None].reciprocal(),
                            current[:, None],
                            torch.minimum(result["support"][:, -2], current)[:, None],
                        ),
                        dim=-1,
                    )
                    for local, index in enumerate(query_ids):
                        produced[index] = (
                            feature[local].cpu().numpy().astype(np.float32, copy=False)
                        )
            rows = [source.row(index) for index in indices]
            values = {
                "features": np.stack([produced[index] for index in indices]),
                "target_phase": np.asarray([row["target_phase"] for row in rows], np.float32),
                "group_id": np.asarray([row["group_id"] for row in rows]),
                "sample_token": np.asarray([row["sample_token"] for row in rows]),
                "sequence_id": np.asarray([row["sequence_id"] for row in rows]),
                "role": np.full(len(rows), ROLE),
            }
            store.write(part, values)
            parts.append(values)
            atomic_write_json(
                store.root / "PROGRESS.json",
                {"completed_rows": stop, "total_rows": source.population_size, "status": "RUNNING"},
            )
    merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    observed_identity = ordered_row_identity_sha256(
        merged["sample_token"].astype(str).tolist(), merged["group_id"].astype(str).tolist()
    )
    if observed_identity != expected_ordered_row_identity_sha256:
        raise ValueError("materialized PAIR rows differ from frozen TRAIN40 order")
    receipt = build_pair_cache_from_features(
        merged.pop("features"),
        merged,
        output,
        modality=modality,
        parents=parent_hashes,
        native_train40_rows_sha256=native_train40_rows_sha256,
    )
    atomic_write_json(
        store.root / "PROGRESS.json",
        {
            "completed_rows": source.population_size,
            "total_rows": source.population_size,
            "status": "COMPLETE",
        },
    )
    return receipt


def materialize_historical_event_pair(
    v12_run: Path,
    output: Path,
    *,
    parents: Mapping[str, str],
    native_train40_rows_sha256: str,
    expected_ordered_row_identity_sha256: str,
) -> dict[str, Any]:
    """Promote the verified V12 A5 133-D bytes into the new fresh-PAIR cache schema."""

    feature_path = v12_run / "PAIR_FEATURES.npz"
    manifest_path = v12_run / "PAIR_FEATURE_MANIFEST.json"
    index_path = v12_run / "TRAIN40_INDEX.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "COMPLETE_VERIFIED"
        or int(manifest.get("row_count", -1)) != 88_744
        or manifest["files"][0]["sha256"] != sha256_file(feature_path)
    ):
        raise ValueError("V12 A5 PAIR feature bytes are incomplete or changed")
    with np.load(feature_path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], np.float32)
        stored_tokens = np.asarray(archive["tokens"]).astype(str)
    with np.load(index_path, allow_pickle=False) as archive:
        tokens = np.asarray(archive["tokens"]).astype(str)
        sequence = np.asarray(archive["sequences"]).astype(str)
        target = np.asarray(archive["phase"], np.float32)
    if not np.array_equal(tokens, stored_tokens) or len(set(tokens.tolist())) != 88_744:
        raise ValueError("V12 A5 PAIR cache and TRAIN40 index token identities differ")
    lineage = {
        **dict(parents),
        "V12_PAIR_FEATURES": sha256_file(feature_path),
        "V12_TRAIN40_INDEX": sha256_file(index_path),
    }
    observed_identity = ordered_row_identity_sha256(tokens.tolist(), sequence.tolist())
    if observed_identity != expected_ordered_row_identity_sha256:
        raise ValueError("historical event PAIR rows differ from frozen TRAIN40 order")
    receipt = build_pair_cache_from_features(
        features,
        {
            "target_phase": target,
            "group_id": sequence,
            "sample_token": tokens,
            "sequence_id": sequence,
            "role": np.full(len(tokens), ROLE),
        },
        output,
        modality="event",
        parents=lineage,
        native_train40_rows_sha256=native_train40_rows_sha256,
    )
    return receipt


def _write_npz_once(path: Path, values: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(values) or any(
                not np.array_equal(archive[name], values[name]) for name in values
            ):
                raise ValueError(f"existing materialization differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    cast(Callable[..., None], np.savez_compressed)(temporary, **values)
    os.replace(temporary, path)


def materialize_rgb_observations(
    source: AllTrainRGBSource,
    a5_model: nn.Module,
    c2f_model: nn.Module,
    pair_model: nn.Module,
    observations_output: Path,
    history_output: Path,
    *,
    parents: Mapping[str, str],
    device: str = "cpu",
    batch_size: int = 32,
    guard: Callable[[], None] | None = None,
    include_targets: bool = True,
) -> dict[str, Any]:
    """Decode each unique real RGB history observation and assemble causal query histories."""

    if device == "cuda" and not torch.cuda.is_available() or batch_size <= 0:
        raise ValueError("invalid RGB observation materialization device or batch size")
    if observations_output.exists() and history_output.exists():
        receipt = _existing_cache(observations_output, "final_refit_rgb_observations_v1", parents)
        history_receipt = json.loads(
            history_output.with_suffix(".json").read_text(encoding="utf-8")
        )
        if (
            history_receipt.get("parents") != dict(parents)
            or history_receipt.get("observations_sha256") != receipt["output_sha256"]
            or history_receipt.get("history_sha256") != sha256_file(history_output)
            or history_receipt.get("targets_included", True) != include_targets
        ):
            raise ValueError("existing RGB history identity differs")
        return {
            **receipt,
            "history": str(history_output.resolve()),
            "history_sha256": history_receipt["history_sha256"],
        }
    observations: dict[str, dict[str, Any]] = {}
    histories: list[list[str]] = []
    query_rows: list[Mapping[str, Any]] = []
    for query_index in range(source.population_size):
        if guard is not None and query_index % 256 == 0:
            guard()
        row = source.row(query_index)
        group = str(row["group_id"])
        role = str(row["historical_role"])
        selected = source.history(query_index)
        if not selected or len(selected) > 8:
            raise ValueError("RGB query has no causal observation or exceeds eight slots")
        identities: list[str] = []
        for item in selected:
            identity = str(item["observation_id"])
            enriched = {**item, "historical_role": role, "group_id": group}
            old = observations.setdefault(identity, enriched)
            if old != enriched:
                raise ValueError("one RGB observation has conflicting group or sensor metadata")
            identities.append(identity)
        histories.append(identities)
        query_rows.append(row)
    ordered = sorted(observations)
    maximum_id = max(map(len, ordered))
    store = FragmentStore(
        observations_output.with_suffix(".parts"),
        {
            "schema": "final_refit_rgb_observation_fragments_v1",
            "queries": source.population_size,
            "observations": len(ordered),
            "parents": dict(sorted(parents.items())),
            "device": device,
            "precision": "float32",
            "targets_are_model_inputs": False,
        },
    )
    a5_model = a5_model.to(device).float().eval().requires_grad_(False)
    c2f_model = c2f_model.to(device).float().eval().requires_grad_(False)
    pair_model = pair_model.to(device).float().eval().requires_grad_(False)
    readers = {
        role: TarFrameReader(source.datasets[role].eap_root) for role in sorted(source.datasets)
    }
    parts: list[dict[str, np.ndarray]] = []
    try:
        with torch.inference_mode():
            for part, start in enumerate(range(0, len(ordered), batch_size)):
                saved = store.read(part)
                if saved is not None:
                    parts.append(saved)
                    continue
                if guard is not None:
                    guard()
                ids = ordered[start : start + batch_size]
                produced: dict[str, dict[str, np.ndarray]] = {}
                grouped: dict[tuple[str, int], list[str]] = {}
                for identity in ids:
                    item = observations[identity]
                    grouped.setdefault(
                        (str(item["historical_role"]), len(item["producer_members"])), []
                    ).append(identity)
                for (role, _), selected_ids in grouped.items():
                    decoded = [
                        decode_query(
                            observations[identity],
                            eap_root=source.datasets[role].eap_root,
                            reader=readers[role],
                        )
                        for identity in selected_ids
                    ]
                    sensor = torch.from_numpy(np.stack([item["rgb"] for item in decoded])).to(
                        device
                    )
                    delta = torch.from_numpy(np.stack([item["delta_t_s"] for item in decoded])).to(
                        device
                    )
                    a5 = producer_features(cast(CausalScaleTTC, a5_model), sensor, delta)
                    c2f = producer_features(cast(CausalScaleTTC, c2f_model), sensor, delta)
                    support = a5["support"]
                    current = support[:, -1]
                    pair_features = pair133(
                        a5["token128"],
                        delta[:, -1],
                        torch.stack((current, torch.minimum(support[:, -2], current)), dim=-1),
                    )
                    pair_phase = emitted_phase(pair_point_phase(pair_model, pair_features))
                    base = raw_rgb_statistics(sensor[:, -1])
                    for local, identity in enumerate(selected_ids):
                        produced[identity] = {
                            "base_statistics2": base[local].cpu().numpy(),
                            "a5_diagnostics3": torch.stack(
                                (a5["flow"][local], a5["margin"][local], a5["log_variance"][local])
                            )
                            .cpu()
                            .numpy(),
                            "c2f_diagnostics3": torch.stack(
                                (
                                    c2f["flow"][local],
                                    c2f["margin"][local],
                                    c2f["log_variance"][local],
                                )
                            )
                            .cpu()
                            .numpy(),
                            "expert_phase": torch.stack(
                                (
                                    a5["point_phase"][local],
                                    c2f["point_phase"][local],
                                    pair_phase[local],
                                )
                            )
                            .cpu()
                            .numpy(),
                        }
                items = [observations[identity] for identity in ids]
                values: dict[str, np.ndarray] = {
                    name: np.stack([produced[identity][name] for identity in ids]).astype(
                        np.float32
                    )
                    for name in (
                        "base_statistics2",
                        "a5_diagnostics3",
                        "c2f_diagnostics3",
                        "expert_phase",
                    )
                }
                values.update(
                    {
                        "observation_id": np.asarray(ids),
                        "anchor_us": np.asarray([item["anchor_us"] for item in items], np.int64),
                        "available_us": np.asarray(
                            [max(item["producer_available_us"]) for item in items], np.int64
                        ),
                        "input_span_us": np.asarray(
                            [
                                int(item["producer_sensor_times_us"][-1])
                                - int(item["producer_sensor_times_us"][0])
                                for item in items
                            ],
                            np.int64,
                        ),
                        "group_id": np.asarray([item["group_id"] for item in items]),
                        "role": np.full(len(items), ROLE),
                    }
                )
                store.write(part, values)
                parts.append(values)
    finally:
        for reader in readers.values():
            reader.close()
    raw = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    receipt = build_observation_cache(
        raw,
        observations_output,
        modality="rgb",
        parents=parents,
    )
    lookup = {identity: index for index, identity in enumerate(raw["observation_id"].astype(str))}
    requested = np.full((source.population_size, 8), "", dtype=f"U{maximum_id}")
    valid = np.zeros((source.population_size, 8), dtype=np.bool_)
    timing = np.zeros((source.population_size, 8, 4), dtype=np.float32)
    experts = np.empty((source.population_size, 3), dtype=np.float32)
    for index, (row, identities) in enumerate(zip(query_rows, histories, strict=True)):
        positions = [lookup[identity] for identity in identities]
        anchors = [int(raw["anchor_us"][position]) for position in positions]
        available = [int(raw["available_us"][position]) for position in positions]
        query_us = int(row["query_time_us"])
        requested[index, -len(identities) :] = identities
        valid[index, -len(identities) :] = True
        timing[index, -len(identities) :] = clock_features(anchors, available, query_us)
        experts[index] = raw["expert_phase"][positions[-1]]
    history_values = {
        "observation_id": requested,
        "timing": timing,
        "valid": valid,
        "expert_phase": experts,
        "sample_token": np.asarray([row["sample_token"] for row in query_rows]),
        "group_id": np.asarray([row["group_id"] for row in query_rows]),
        "role": np.full(source.population_size, ROLE),
    }
    if include_targets:
        history_values["target_phase"] = np.asarray(
            [row["target_phase"] for row in query_rows], np.float32
        )
    _write_npz_once(history_output, history_values)
    atomic_write_json(
        history_output.with_suffix(".json"),
        {
            "schema": "final_refit_raw_history_v1",
            "status": "COMPLETE",
            "role": ROLE,
            "modality": "rgb",
            "queries": source.population_size,
            "observations": len(ordered),
            "groups": len(set(history_values["group_id"].tolist())),
            "parents": dict(sorted(parents.items())),
            "observations_sha256": receipt["output_sha256"],
            "history_sha256": sha256_file(history_output),
            "causal_clocks_derived_from_integer_manifest_fields": True,
            "optimizer_updates": 0,
            "targets_included": include_targets,
        },
    )
    return {
        **receipt,
        "history": str(history_output.resolve()),
        "history_sha256": sha256_file(history_output),
    }


def materialize_context_cache(
    observations_path: Path,
    history_path: Path,
    output: Path,
    *,
    modality: str,
    parents: Mapping[str, str],
    native_train40_rows_sha256: str,
    expected_ordered_row_identity_sha256: str,
) -> dict[str, Any]:
    """Fit the all-TRAIN normalizer, then gather and seal the temporal cache."""

    if output.exists():
        return _existing_cache(
            output,
            "final_refit_context_cache_v1",
            parents,
            modality=modality,
            native_train40_rows_sha256=native_train40_rows_sha256,
            expected_ordered_row_identity_sha256=expected_ordered_row_identity_sha256,
        )
    with np.load(observations_path, allow_pickle=False) as archive:
        observations = {name: archive[name] for name in archive.files}
    with np.load(history_path, allow_pickle=False) as archive:
        history = {name: archive[name] for name in archive.files}
    observation_receipt = validate_cache_receipt(
        observations_path, f"final_refit_{modality}_observations_v1"
    )
    if observation_receipt["parents"] != dict(sorted(parents.items())):
        raise ValueError("observation cache parent bindings differ from context ancestry")
    _validate_raw_history_receipt(
        history_path,
        modality=modality,
        parents=parents,
        observation_receipt=observation_receipt,
        history=history,
    )
    population = normalizer_population_binding(
        observations,
        history,
        native_train40_rows_sha256=native_train40_rows_sha256,
        observation_receipt_sha256=sha256_file(observations_path.with_suffix(".json")),
    )
    if population["ordered_row_identity_sha256"] != expected_ordered_row_identity_sha256:
        raise ValueError("context rows differ from the externally frozen TRAIN40 inventory")
    normalizer = output.with_name(f"{output.stem}_NORMALIZER.json")
    if not normalizer.exists():
        fit_train40_normalizer(
            observations,
            normalizer,
            modality=modality,
            parents=parents,
            population_binding=population,
        )
    return build_context_cache(
        observations,
        history,
        normalizer,
        output,
        modality=modality,
        parents=parents,
        population_binding=population,
    )


def materialize_phase_states(
    event_context: Path,
    rgb_context: Path,
    event_model: nn.Module,
    rgb_model: nn.Module,
    output: Path,
    *,
    parents: Mapping[str, str],
    native_train40_rows_sha256: str,
    expected_ordered_row_identity_sha256: str,
    device: str = "cpu",
    batch_size: int = 128,
) -> dict[str, Any]:
    """Run completed temporal heads and seal exact V16 PHASE_ONLY states."""

    if output.exists():
        return _existing_cache(
            output,
            "final_refit_phase_cache_v1",
            parents,
            native_train40_rows_sha256=native_train40_rows_sha256,
            expected_ordered_row_identity_sha256=expected_ordered_row_identity_sha256,
        )
    with np.load(event_context, allow_pickle=False) as archive:
        event = {name: archive[name] for name in archive.files}
    with np.load(rgb_context, allow_pickle=False) as archive:
        rgb = {name: archive[name] for name in archive.files}
    observed_identity = ordered_row_identity_sha256(
        np.asarray(event["sample_token"]).astype(str).tolist(),
        np.asarray(event["group_id"]).astype(str).tolist(),
    )
    if observed_identity != expected_ordered_row_identity_sha256:
        raise ValueError("phase inputs differ from frozen TRAIN40 order")
    normalizers = {
        "event": sha256_file(event_context.with_name(f"{event_context.stem}_NORMALIZER.json")),
        "rgb": sha256_file(rgb_context.with_name(f"{rgb_context.stem}_NORMALIZER.json")),
    }
    return build_phase_cache(
        event,
        rgb,
        event_model,
        rgb_model,
        output,
        parents=parents,
        normalizers=normalizers,
        native_train40_rows_sha256=native_train40_rows_sha256,
        device=device,
        batch_size=batch_size,
    )


def pipeline_status(run_root: Path) -> dict[str, Any]:
    """Report exact stage availability without executing models or reading private test data."""

    run_root = run_root.resolve()
    expected = {
        "PAIR_E_MATCHED": run_root / "features/PAIR_E_MATCHED.npz",
        "PAIR_R": run_root / "features/PAIR_R.npz",
        "E_CTX_MATCHED": run_root / "features/E_CTX_MATCHED.npz",
        "R_CTX": run_root / "features/R_CTX.npz",
        "PHASE_ONLY_EVENT_S7_TRAIN40": run_root / "features/PHASE_ONLY_EVENT_S7_TRAIN40.npz",
    }
    schemas = {
        "PAIR_E_MATCHED": "final_refit_pair_cache_v1",
        "PAIR_R": "final_refit_pair_cache_v1",
        "E_CTX_MATCHED": "final_refit_context_cache_v1",
        "R_CTX": "final_refit_context_cache_v1",
        "PHASE_ONLY_EVENT_S7_TRAIN40": "final_refit_phase_cache_v1",
    }
    stages: dict[str, Any] = {}
    for name, path in expected.items():
        try:
            receipt = validate_cache_receipt(path, schemas[name])
            stages[name] = {
                "status": "COMPLETE",
                "path": str(path),
                "sha256": receipt["output_sha256"],
            }
        except FileNotFoundError:
            stages[name] = {"status": "MISSING", "path": str(path)}
        except ValueError as error:
            stages[name] = {"status": "INVALID", "path": str(path), "reason": str(error)}
    return {
        "schema": "final_refit_materialization_status_v1",
        "run_root": str(run_root),
        "stages": stages,
        "complete": all(value["status"] == "COMPLETE" for value in stages.values()),
        "optimizer_updates": 0,
    }


def _json_mapping(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("parent JSON must be a mapping")
    return {str(name): str(digest) for name, digest in value.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    event_pair = commands.add_parser("event-pair")
    event_pair.add_argument("--v12-run", type=Path, required=True)
    event_pair.add_argument("--output", type=Path, required=True)
    event_pair.add_argument("--parents", type=Path, required=True)
    event_pair.add_argument("--native-train40-rows-sha256", required=True)
    event_pair.add_argument("--ordered-row-identity-sha256", required=True)
    rgb_pair = commands.add_parser("rgb-pair")
    rgb_pair.add_argument("--source", type=Path, required=True)
    rgb_pair.add_argument("--teacher", type=Path, required=True)
    rgb_pair.add_argument("--config", type=Path, required=True)
    rgb_pair.add_argument("--recipe", type=Path, required=True)
    rgb_pair.add_argument("--run-root", type=Path, required=True)
    rgb_pair.add_argument("--output", type=Path, required=True)
    rgb_pair.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    rgb_observations = commands.add_parser("rgb-observations")
    for option in ("source", "teacher", "config", "recipe", "run-root"):
        rgb_observations.add_argument(f"--{option}", type=Path, required=True)
    rgb_observations.add_argument("--observations-output", type=Path, required=True)
    rgb_observations.add_argument("--history-output", type=Path, required=True)
    rgb_observations.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    context = commands.add_parser("context")
    context.add_argument("--observations", type=Path, required=True)
    context.add_argument("--history", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    context.add_argument("--modality", choices=("event", "rgb"), required=True)
    context.add_argument("--parents", type=Path, required=True)
    context.add_argument("--native-train40-rows-sha256", required=True)
    context.add_argument("--ordered-row-identity-sha256", required=True)
    observations = commands.add_parser("observations")
    observations.add_argument("--raw-blocks", type=Path, required=True)
    observations.add_argument("--output", type=Path, required=True)
    observations.add_argument("--modality", choices=("event", "rgb"), required=True)
    observations.add_argument("--parents", type=Path, required=True)
    phase = commands.add_parser("phase")
    phase.add_argument("--recipe", type=Path, required=True)
    phase.add_argument("--run-root", type=Path, required=True)
    phase.add_argument("--event-context", type=Path, required=True)
    phase.add_argument("--rgb-context", type=Path, required=True)
    phase.add_argument("--output", type=Path, required=True)
    phase.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.command == "status":
        result = pipeline_status(args.run_root)
    elif args.command == "event-pair":
        result = materialize_historical_event_pair(
            args.v12_run.resolve(strict=True),
            args.output.resolve(),
            parents=_json_mapping(args.parents.resolve(strict=True)),
            native_train40_rows_sha256=args.native_train40_rows_sha256,
            expected_ordered_row_identity_sha256=args.ordered_row_identity_sha256,
        )
    elif args.command == "context":
        result = materialize_context_cache(
            args.observations.resolve(strict=True),
            args.history.resolve(strict=True),
            args.output.resolve(),
            modality=args.modality,
            parents=_json_mapping(args.parents.resolve(strict=True)),
            native_train40_rows_sha256=args.native_train40_rows_sha256,
            expected_ordered_row_identity_sha256=args.ordered_row_identity_sha256,
        )
    elif args.command == "observations":
        with np.load(args.raw_blocks.resolve(strict=True), allow_pickle=False) as archive:
            raw_blocks = {name: archive[name] for name in archive.files}
        result = build_observation_cache(
            raw_blocks,
            args.output.resolve(),
            modality=args.modality,
            parents=_json_mapping(args.parents.resolve(strict=True)),
        )
    elif args.command == "phase":
        recipe = load_recipe(args.recipe.resolve(strict=True))
        run_root = args.run_root.resolve(strict=True)
        event_stage = stage(recipe, "E_CTX_MATCHED")
        rgb_stage = stage(recipe, "R_CTX")
        event_fit = run_root / "fits/E_CTX_MATCHED"
        rgb_fit = run_root / "fits/R_CTX"
        event_model = load_cached_endpoint(event_fit, event_stage, torch.device(args.device))
        rgb_model = load_cached_endpoint(rgb_fit, rgb_stage, torch.device(args.device))
        result = materialize_phase_states(
            args.event_context.resolve(strict=True),
            args.rgb_context.resolve(strict=True),
            event_model,
            rgb_model,
            args.output.resolve(),
            parents=bind_files(
                {
                    "E_CTX_MATCHED": event_fit / "checkpoint_last.pt",
                    "R_CTX": rgb_fit / "checkpoint_last.pt",
                    "FINAL_REFIT_RECIPE": args.recipe.resolve(strict=True),
                }
            ),
            native_train40_rows_sha256=recipe.native_train40_rows_sha256,
            expected_ordered_row_identity_sha256=recipe.ordered_row_identity_sha256,
            device=args.device,
        )
    elif args.command == "rgb-observations":
        recipe = load_recipe(args.recipe.resolve(strict=True))
        run_root = args.run_root.resolve(strict=True)
        source = load_all_train_source(
            args.source.resolve(strict=True), args.teacher.resolve(strict=True)
        )
        try:
            models: dict[str, nn.Module] = {}
            parent_paths: dict[str, Path] = {}
            for fit_id in ("R_A5", "R_C2F"):
                producer_recipe = build_rgb_recipe(
                    args.config.resolve(strict=True),
                    fit_id,
                    args.source.resolve(strict=True),
                    source,
                )
                fit = run_root / "fits" / fit_id
                models[fit_id] = load_rgb_endpoint(fit, producer_recipe, device=args.device)
                parent_paths[fit_id] = fit / "checkpoint_last.pt"
            pair_fit = run_root / "fits/PAIR_R"
            models["PAIR_R"] = load_cached_endpoint(
                pair_fit, stage(recipe, "PAIR_R"), torch.device(args.device)
            )
            parent_paths["PAIR_R"] = pair_fit / "checkpoint_last.pt"
            result = materialize_rgb_observations(
                source,
                models["R_A5"],
                models["R_C2F"],
                models["PAIR_R"],
                args.observations_output.resolve(),
                args.history_output.resolve(),
                parents=bind_files(parent_paths),
                device=args.device,
            )
        finally:
            source.close()
    else:
        recipe = load_recipe(args.recipe.resolve(strict=True))
        source: AllTrainRGBSource = load_all_train_source(
            args.source.resolve(strict=True), args.teacher.resolve(strict=True)
        )
        try:
            producer_recipe = build_rgb_recipe(
                args.config.resolve(strict=True), "R_A5", args.source.resolve(strict=True), source
            )
            fit = args.run_root.resolve() / "fits/R_A5"
            model = load_rgb_endpoint(fit, producer_recipe, device=args.device)
            parents = bind_files(
                {
                    "R_A5": fit / "checkpoint_last.pt",
                    "FINAL_REFIT_RECIPE": args.recipe.resolve(strict=True),
                }
            )
            if stage(recipe, "PAIR_R").cache_schema != "final_refit_pair_cache_v1":
                raise ValueError("registered PAIR_R cache schema differs")
            result = materialize_query_pair_features(
                source,
                model,
                args.output.resolve(),
                modality="rgb",
                parents=parents,
                native_train40_rows_sha256=recipe.native_train40_rows_sha256,
                expected_ordered_row_identity_sha256=recipe.ordered_row_identity_sha256,
                device=args.device,
            )
        finally:
            source.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FragmentStore",
    "materialize_context_cache",
    "materialize_historical_event_pair",
    "materialize_phase_states",
    "materialize_query_pair_features",
    "materialize_rgb_observations",
    "pipeline_status",
]
