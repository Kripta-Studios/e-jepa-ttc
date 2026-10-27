"""Durable TRAIN40 event-history materialization for the final V16 refit.

The completed V12 H8 cache remains the byte authority for PHASE17 columns
0:10 (raw event statistics, A5/C2F diagnostics and their phases).  This module
replays the exact query-conditioned event crop only to obtain the A5 token and
support required by the newly trained PAIR_E.  It then emits the fresh PAIR
phase and lets :func:`build_observation_cache` reconstruct columns 10:17.

No optimizer step or test input is reachable from this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from e_jepa_ttc.simplex_t.context_dedup import ContextSource, observation_key
from e_jepa_ttc.simplex_t.phase import emitted_phase
from e_jepa_ttc.simplex_t.query_context import QueryContextInput
from operational.rgb_port.accounting import atomic_write_json
from operational.rgb_port.infer_experts import clock_features
from operational.rgb_port.train_producers import producer_features

from .event_parity import load_v12_event_producer, sha256_file
from .features import ROLE, build_observation_cache
from .materialize import FragmentStore, _write_npz_once
from .recipe import load_recipe, stage
from .training import load_cached_endpoint

ROWS = 88_744
SLOTS = 8
HISTORICAL_PREFIX = 10
PHASE17_WIDTH = 17


@dataclass(frozen=True)
class EventHistorySources:
    """Validated immutable metadata needed to replay TRAIN40 histories."""

    index: Mapping[str, np.ndarray]
    frames: Mapping[tuple[str, str], Mapping[str, Any]]
    raw_pins: Mapping[str, Mapping[str, Any]]
    bounds: Mapping[str, tuple[int, int]]
    producer_family_sha256: str
    preprocessing_sha256: str


@dataclass(frozen=True)
class QueryPlan:
    """One label-free, exact-ROI event replay request."""

    row: int
    sequence: str
    path: Path
    windows: np.ndarray
    square: np.ndarray
    valid: np.ndarray
    positions: np.ndarray
    observation_ids: tuple[str, ...]
    delta_seconds: float
    anchor_us: int
    available_us: int
    input_span_us: int


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return cast(dict[str, Any], value)


def _historical_pair_input(output: Mapping[str, Tensor], delta_seconds: Tensor) -> Tensor:
    """Reproduce the admitted V12 133-D PAIR input, including its log epsilon."""

    token = output["token128"].float()
    support = output["support"].float()
    if token.ndim != 2 or token.shape[1] != 128 or support.shape != (len(token), 3):
        raise ValueError("A5 history output differs from the admitted T3 producer contract")
    if delta_seconds.shape != (len(token),) or bool((delta_seconds <= 0).any()):
        raise ValueError("history delta must be one positive FP32 scalar per observation")
    current = support[:, -1]
    return torch.cat(
        (
            token,
            torch.stack(
                (
                    delta_seconds,
                    torch.log(delta_seconds + 1e-8),
                    delta_seconds.reciprocal(),
                ),
                -1,
            ),
            current[:, None],
            torch.minimum(support[:, -2], current)[:, None],
        ),
        -1,
    ).float()


def merge_fresh_pair_phase(historical_prefix: np.ndarray, pair_phase: np.ndarray) -> np.ndarray:
    """Combine immutable columns 0:10 with fresh PAIR and rebuilt differences."""

    prefix = np.asarray(historical_prefix, dtype=np.float32)
    phase = np.asarray(pair_phase, dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != HISTORICAL_PREFIX:
        raise ValueError("historical event prefix must be finite FP32 [N,10]")
    if (
        phase.shape != (len(prefix),)
        or not np.isfinite(prefix).all()
        or not np.isfinite(phase).all()
    ):
        raise ValueError("event history prefix and PAIR phase must be finite and aligned")
    expert = np.column_stack((prefix[:, 8:10], phase)).astype(np.float32, copy=False)
    signed = np.stack(
        (expert[:, 2] - expert[:, 0], expert[:, 2] - expert[:, 1], expert[:, 1] - expert[:, 0]),
        axis=-1,
    ).astype(np.float32, copy=False)
    result = np.concatenate((prefix, phase[:, None], signed, np.abs(signed)), axis=-1)
    if result.shape != (len(prefix), PHASE17_WIDTH) or not np.isfinite(result).all():
        raise AssertionError("rebuilt event PHASE17 contract drifted")
    return np.ascontiguousarray(result, dtype=np.float32)


def causal_history_mask(
    anchors_us: Sequence[int], *, query_us: int, input_span_us: int
) -> np.ndarray:
    """Select the chronological suffix whose complete sensor input fits in 650 ms."""

    anchors = np.asarray(anchors_us, dtype=np.int64)
    if (
        anchors.ndim != 1
        or len(anchors) < 1
        or len(anchors) > SLOTS
        or np.any(np.diff(anchors) <= 0)
        or input_span_us <= 0
        or np.any(anchors > query_us)
    ):
        raise ValueError("event history clocks are invalid")
    result = query_us - anchors + input_span_us <= 650_000
    if not bool(result[-1]) or np.any(result[:-1] & ~result[1:]):
        raise ValueError("650 ms event history selection is not suffix-causal")
    return result


def compact_referenced_observations(
    observation_source: Mapping[str, np.ndarray],
    requested: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Keep exactly the valid history observations in their historical row order."""

    identities = np.asarray(observation_source["observation_id"]).astype(str)
    requested_ids = np.asarray(requested).astype(str)
    valid_mask = np.asarray(valid, dtype=np.bool_)
    if requested_ids.ndim != 2 or requested_ids.shape != valid_mask.shape:
        raise ValueError("event history observation IDs and validity mask differ")
    if len(set(identities.tolist())) != len(identities):
        raise ValueError("generated event observation identities must be unique")
    if any(len(np.asarray(value)) != len(identities) for value in observation_source.values()):
        raise ValueError("generated event observation arrays are not row-aligned")
    referenced = set(requested_ids[valid_mask].tolist())
    if "" in referenced:
        raise ValueError("valid event history references an empty observation identity")
    missing = sorted(referenced - set(identities.tolist()))
    if missing:
        raise ValueError(f"event history references missing generated observations: {missing[:3]}")
    keep = np.asarray([identity in referenced for identity in identities], dtype=np.bool_)
    compacted = {name: np.asarray(value)[keep] for name, value in observation_source.items()}
    if set(np.asarray(compacted["observation_id"]).astype(str).tolist()) != referenced:
        raise AssertionError("event observation compaction did not preserve the referenced set")
    return compacted


def _load_sources(v12_run: Path, raw_root: Path) -> EventHistorySources:
    """Load only metadata and validate the original source bindings."""

    import h5py
    import pyarrow.dataset as ds

    from e_jepa_ttc.data.eap import _require_hdf5plugin

    _require_hdf5plugin()

    index_path = v12_run / "TRAIN40_INDEX.npz"
    with np.load(index_path, allow_pickle=False) as archive:
        names = (
            "tokens",
            "sequences",
            "windows_us",
            "square_xyxy",
            "delta_t_s",
            "rgb_members",
            "phase",
        )
        index = {name: archive[name] for name in names}
    if any(len(value) != ROWS for value in index.values()):
        raise ValueError("TRAIN40 index population differs from the admitted 88,744 rows")
    audit = _read_json(v12_run / "DATA_AUDIT.json")
    media_path = raw_root / "data/train.parquet"
    media = (
        ds.dataset(media_path)
        .to_table(
            columns=[
                "sequence_id",
                "rgb_member_path",
                "rgb_exposure_start_timestamp_us",
                "rgb_exposure_end_timestamp_us",
            ],
            filter=ds.field("sequence_id").isin(audit["sequences"]),
        )
        .to_pylist()
    )
    frames = {(str(row["sequence_id"]), str(row["rgb_member_path"])): row for row in media}
    if len(frames) != len(media):
        raise ValueError("ambiguous TRAIN exposure reference")
    pins = {
        str(item["sequence_id"]): item for item in _read_json(v12_run / "RAW_PLAN.json")["files"]
    }
    bounds: dict[str, tuple[int, int]] = {}
    for sequence in audit["sequences"]:
        path = raw_root / "data/train" / sequence / "events.h5"
        with h5py.File(path, "r") as handle:
            timestamps = handle["events/t"]
            if not isinstance(timestamps, h5py.Dataset):
                raise ValueError("raw event timestamps must be an HDF5 dataset")
            bounds[str(sequence)] = (int(timestamps[0]), int(timestamps[-1]))
    h8_manifest = _read_json(v12_run / "H8_FEATURE_MANIFEST.json")
    parents = h8_manifest.get("parents")
    if not isinstance(parents, dict):
        raise ValueError("historical H8 parent identity is missing")
    family_sha = hashlib.sha256(json.dumps(parents, sort_keys=True).encode()).hexdigest()
    return EventHistorySources(
        index=index,
        frames=frames,
        raw_pins=pins,
        bounds=bounds,
        producer_family_sha256=family_sha,
        preprocessing_sha256=sha256_file(v12_run / "PREPARE_FREEZE.json"),
    )


def _query_plan(
    sources: EventHistorySources, raw_root: Path, old_history: np.ndarray, row: int
) -> QueryPlan:
    index = sources.index
    sequence = str(index["sequences"][row])
    windows = np.asarray(index["windows_us"][row], dtype=np.int64)
    start_bound, end_bound = sources.bounds[sequence]
    endpoints = [
        sources.frames[(sequence, str(member))] for member in index["rgb_members"][row]
    ]
    exposure_starts = [int(frame["rgb_exposure_start_timestamp_us"]) for frame in endpoints]
    if exposure_starts != windows[1:, 1].tolist():
        raise ValueError("TRAIN exposure and event clock mapping differs")
    available = max(int(frame["rgb_exposure_end_timestamp_us"]) for frame in endpoints)
    if available < int(windows[-1, 1]) or int(windows.max()) > end_bound:
        raise ValueError("current exposure or complete raw support is missing")
    current = QueryContextInput(
        query_token=f"{row}:{index['tokens'][row]}",
        producer_family_sha256=sources.producer_family_sha256,
        anchor_us=int(windows[-1, 1]),
        roi_available_us=available,
        allowed_cutoff_us=available,
        source_start_us=start_bound,
        windows_us=(
            (int(windows[0, 0]), int(windows[0, 1])),
            (int(windows[1, 0]), int(windows[1, 1])),
            (int(windows[2, 0]), int(windows[2, 1])),
        ),
        square_xyxy=(
            float(index["square_xyxy"][row, 0]),
            float(index["square_xyxy"][row, 1]),
            float(index["square_xyxy"][row, 2]),
            float(index["square_xyxy"][row, 3]),
        ),
    )
    source = ContextSource(
        sequence,
        str(sources.raw_pins[sequence]["sha256"]),
        sources.preprocessing_sha256,
        current,
    )
    valid = windows.min() - np.arange(7, -1, -1, dtype=np.int64) * 50_000 >= start_bound
    if not bool(valid[-1]):
        raise ValueError("current TRAIN context lacks causal raw support")
    identities = tuple(
        observation_key(source, int(lag))
        for lag, present in zip(range(350_000, -1, -50_000), valid, strict=True)
        if present
    )
    positions = np.asarray(old_history[row][valid], dtype=np.int64)
    if len(positions) != len(identities) or np.any(positions < 0):
        raise ValueError("historical H8 indices differ from reconstructed valid slots")
    return QueryPlan(
        row=row,
        sequence=sequence,
        path=raw_root / "data/train" / sequence / "events.h5",
        windows=windows,
        square=np.asarray(index["square_xyxy"][row], dtype=np.float32),
        valid=valid,
        positions=positions,
        observation_ids=identities,
        delta_seconds=float(index["delta_t_s"][row]),
        anchor_us=int(windows[-1, 1]),
        available_us=available,
        input_span_us=int(windows[-1, 1] - windows[0, 0]),
    )


def _plans(
    sources: EventHistorySources, raw_root: Path, old_history: np.ndarray
) -> Iterator[QueryPlan]:
    for row in range(ROWS):
        yield _query_plan(sources, raw_root, old_history, row)


def _fresh_pair_phases(
    plan: QueryPlan,
    reader_pool: ReaderPool,
    a5_model: CausalScaleTTC,
    pair_model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Replay one bounded query and return its valid eight-slot fresh PAIR phases."""

    from e_jepa_ttc.efficient_context.mapped_union import encode_union

    tensor = encode_union(
        reader_pool.get(plan.path),
        plan.windows,
        np.arange(15, -1, -1, dtype=np.int64) * 50_000,
        np.concatenate((np.zeros(8, dtype=np.bool_), plan.valid)),
        tuple(float(value) for value in plan.square),
        sequence_id=plan.sequence,
        roi_size=128,
        event_pixel_diff=5,
    ).to(device=device, dtype=torch.float32)
    delta = torch.full((16, 2), plan.delta_seconds, dtype=torch.float32, device=device)
    output = producer_features(a5_model, tensor, delta)
    elapsed = delta[:, -1]
    features = _historical_pair_input(output, elapsed)
    with torch.inference_mode():
        raw_phase = pair_model(PairFeatureBatch(features))
        phases = emitted_phase(raw_phase).float().cpu().numpy()[-SLOTS:]
    return np.ascontiguousarray(phases[plan.valid], dtype=np.float32)


def _validate_historical_h8(v12_run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = _read_json(v12_run / "H8_FEATURE_MANIFEST.json")
    path = v12_run / "H8_FEATURES.npz"
    if (
        manifest.get("status") != "COMPLETE_VERIFIED"
        or int(manifest.get("row_count", -1)) != ROWS
        or manifest.get("files", [{}])[0].get("sha256") != sha256_file(path)
    ):
        raise ValueError("historical H8 cache is incomplete or changed")
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        history = np.asarray(archive["history"], dtype=np.int64)
        anchors = np.asarray(archive["anchor_us"], dtype=np.int64)
        available = np.asarray(archive["available_us"], dtype=np.int64)
    if (
        features.ndim != 2
        or features.shape[1] != PHASE17_WIDTH
        or history.shape != (ROWS, SLOTS)
        or anchors.shape != (len(features),)
        or available.shape != (len(features),)
        or not np.isfinite(features[:, :HISTORICAL_PREFIX]).all()
    ):
        raise ValueError("historical H8 array schema differs")
    return features[:, :HISTORICAL_PREFIX].copy(), history, anchors, available


def materialize_event_history(
    *,
    v12_run: Path,
    raw_root: Path,
    a5_model: CausalScaleTTC,
    pair_model: nn.Module,
    observations_output: Path,
    history_output: Path,
    parents: Mapping[str, str],
    device_name: str = "cuda",
    queries_per_fragment: int = 64,
) -> dict[str, Any]:
    """Build final TRAIN40 event observations/history with durable query fragments."""

    if queries_per_fragment < 1 or device_name not in {"cpu", "cuda"}:
        raise ValueError("invalid event-history materialization arguments")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA event-history materialization requested but unavailable")
    prefix, old_history, old_anchors, old_available = _validate_historical_h8(v12_run)
    sources = _load_sources(v12_run, raw_root)
    parent_hashes = dict(sorted((str(k), str(v)) for k, v in parents.items()))
    historical_parents = _read_json(v12_run / "H8_FEATURE_MANIFEST.json")["parents"]
    if (
        parent_hashes.get("E_A5") != historical_parents["a5"]["checkpoint_sha256"]
        or parent_hashes.get("E_C2F") != historical_parents["c2f"]["checkpoint_sha256"]
    ):
        raise ValueError("historical PHASE17 prefix ancestry differs from supplied event parents")
    store = FragmentStore(
        observations_output.with_suffix(".parts"),
        {
            "schema": "final_refit_event_history_fragments_v1",
            "population": ROWS,
            "historical_h8_sha256": sha256_file(v12_run / "H8_FEATURES.npz"),
            "parents": parent_hashes,
            "precision": "float32",
            "device": device_name,
            "queries_per_fragment": queries_per_fragment,
            "query_conditioned_roi": True,
            "optimizer_updates": 0,
        },
    )
    a5_model = cast(CausalScaleTTC, a5_model.to(device).float().eval().requires_grad_(False))
    pair_model = pair_model.to(device).float().eval().requires_grad_(False)
    reader_pool = ReaderPool()
    parts: list[dict[str, np.ndarray]] = []
    try:
        with torch.inference_mode():
            for part, start in enumerate(range(0, ROWS, queries_per_fragment)):
                saved = store.read(part)
                if saved is not None:
                    parts.append(saved)
                    continue
                stop = min(start + queries_per_fragment, ROWS)
                positions: list[np.ndarray] = []
                phases: list[np.ndarray] = []
                for row in range(start, stop):
                    plan = _query_plan(sources, raw_root, old_history, row)
                    positions.append(plan.positions)
                    phases.append(
                        _fresh_pair_phases(plan, reader_pool, a5_model, pair_model, device)
                    )
                values = {
                    "position": np.concatenate(positions).astype(np.int64, copy=False),
                    "pair_phase": np.concatenate(phases).astype(np.float32, copy=False),
                }
                store.write(part, values)
                parts.append(values)
                atomic_write_json(
                    store.root / "PROGRESS.json",
                    {"status": "RUNNING", "completed_queries": stop, "total_queries": ROWS},
                )
    finally:
        reader_pool.close()
    pair_phase = np.full(len(prefix), np.nan, dtype=np.float32)
    for values in parts:
        for position, value in zip(values["position"], values["pair_phase"], strict=True):
            index = int(position)
            if np.isfinite(pair_phase[index]) and pair_phase[index] != value:
                raise ValueError(
                    "replayed duplicate event observation produced a different PAIR phase"
                )
            pair_phase[index] = value
    if not np.isfinite(pair_phase).all():
        raise ValueError("event history replay did not cover every historical observation")
    rebuilt = merge_fresh_pair_phase(prefix, pair_phase)
    identities = np.full(len(prefix), "", dtype="U64")
    groups = np.full(len(prefix), "", dtype="U64")
    input_spans = np.full(len(prefix), -1, dtype=np.int64)
    requested = np.full((ROWS, SLOTS), "", dtype="U64")
    valid = np.zeros((ROWS, SLOTS), dtype=np.bool_)
    timing = np.zeros((ROWS, SLOTS, 4), dtype=np.float32)
    for plan in _plans(sources, raw_root, old_history):
        for position, identity in zip(plan.positions, plan.observation_ids, strict=True):
            index = int(position)
            if identities[index] and identities[index] != identity:
                raise ValueError("historical H8 position maps to multiple observation identities")
            identities[index] = identity
            groups[index] = plan.sequence
            if input_spans[index] not in {-1, plan.input_span_us}:
                raise ValueError("historical observation maps to multiple input spans")
            input_spans[index] = plan.input_span_us
        columns = np.flatnonzero(plan.valid)
        anchors = [int(old_anchors[position]) for position in plan.positions]
        arrivals = [int(old_available[position]) for position in plan.positions]
        if anchors != [plan.anchor_us - (SLOTS - 1 - column) * 50_000 for column in columns]:
            raise ValueError("reconstructed observation anchors differ from historical H8")
        if any(arrival != plan.available_us for arrival in arrivals):
            raise ValueError("reconstructed ROI availability differs from historical H8")
        causal = causal_history_mask(
            anchors,
            query_us=plan.available_us,
            input_span_us=plan.input_span_us,
        )
        kept_columns = columns[causal]
        kept_ids = np.asarray(plan.observation_ids)[causal]
        kept_anchors = np.asarray(anchors, dtype=np.int64)[causal].tolist()
        kept_arrivals = np.asarray(arrivals, dtype=np.int64)[causal].tolist()
        requested[plan.row, kept_columns] = kept_ids
        valid[plan.row, kept_columns] = True
        timing[plan.row, kept_columns] = clock_features(
            kept_anchors, kept_arrivals, plan.available_us
        )
    if np.any(identities == "") or np.any(groups == ""):
        raise ValueError("historical observation identity reconstruction is incomplete")
    observation_source = {
        "observation_id": identities,
        "base_statistics2": rebuilt[:, :2],
        "a5_diagnostics3": rebuilt[:, 2:5],
        "c2f_diagnostics3": rebuilt[:, 5:8],
        "expert_phase": rebuilt[:, 8:11],
        "anchor_us": old_anchors,
        "available_us": old_available,
        "input_span_us": input_spans,
        "group_id": groups,
        "role": np.full(len(prefix), ROLE),
    }
    compacted_observations = compact_referenced_observations(
        observation_source, requested, valid
    )
    receipt = build_observation_cache(
        compacted_observations,
        observations_output,
        modality="event",
        parents=parent_hashes,
    )
    history_values = {
        "observation_id": requested,
        "timing": timing,
        "valid": valid,
        "expert_phase": rebuilt[old_history[:, -1], 8:11],
        "target_phase": np.asarray(sources.index["phase"], dtype=np.float32),
        "sample_token": np.asarray(sources.index["tokens"]).astype(str),
        "group_id": np.asarray(sources.index["sequences"]).astype(str),
        "role": np.full(ROWS, ROLE),
    }
    _write_npz_once(history_output, history_values)
    history_receipt = {
        "schema": "final_refit_raw_history_v1",
        "status": "COMPLETE",
        "role": ROLE,
        "modality": "event",
        "queries": ROWS,
        "observations": len(compacted_observations["observation_id"]),
        "historical_observations_generated": len(prefix),
        "referenced_observations_retained": len(compacted_observations["observation_id"]),
        "pruned_unreferenced_observations": (
            len(prefix) - len(compacted_observations["observation_id"])
        ),
        "groups": len(set(groups.tolist())),
        "parents": parent_hashes,
        "observations_sha256": receipt["output_sha256"],
        "history_sha256": sha256_file(history_output),
        "historical_columns_reused": list(range(HISTORICAL_PREFIX)),
        "fresh_columns": list(range(HISTORICAL_PREFIX, PHASE17_WIDTH)),
        "query_conditioned_roi": True,
        "total_information_budget_us": 650_000,
        "optimizer_updates": 0,
    }
    atomic_write_json(history_output.with_suffix(".json"), history_receipt)
    atomic_write_json(
        store.root / "PROGRESS.json",
        {"status": "COMPLETE", "completed_queries": ROWS, "total_queries": ROWS},
    )
    return {
        **receipt,
        "history": str(history_output.resolve()),
        "history_sha256": history_receipt["history_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12-run", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--observations-output", type=Path, required=True)
    parser.add_argument("--history-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--queries-per-fragment", type=int, default=64)
    args = parser.parse_args(argv)
    v12_run = args.v12_run.resolve(strict=True)
    protocol = _read_json(v12_run / "TRAINING_PROTOCOL.json")
    a5, a5_identity = load_v12_event_producer(v12_run, protocol, "a5")
    recipe = load_recipe(args.recipe.resolve(strict=True))
    if recipe.external_parent_sha256["E_A5"] != a5_identity["checkpoint_sha256"]:
        raise ValueError("recipe E_A5 identity differs from the admitted V12 endpoint")
    h8_manifest = _read_json(v12_run / "H8_FEATURE_MANIFEST.json")
    if recipe.external_parent_sha256["E_C2F"] != h8_manifest["parents"]["c2f"]["checkpoint_sha256"]:
        raise ValueError("historical prefix C2F identity differs from the frozen recipe")
    pair_fit = args.run_root.resolve() / "fits/PAIR_E_MATCHED"
    pair = load_cached_endpoint(
        pair_fit, stage(recipe, "PAIR_E_MATCHED"), torch.device(args.device)
    )
    pair_receipt = _read_json(pair_fit / "CHECKPOINT_RECEIPT.json")
    pair_checkpoint = Path(str(pair_receipt["checkpoint_path"])).resolve(strict=True)
    parents = {
        "E_A5": str(a5_identity["checkpoint_sha256"]),
        "E_C2F": str(recipe.external_parent_sha256["E_C2F"]),
        "PAIR_E_MATCHED": sha256_file(pair_checkpoint),
    }
    result = materialize_event_history(
        v12_run=v12_run,
        raw_root=args.raw_root.resolve(strict=True),
        a5_model=a5,
        pair_model=pair,
        observations_output=args.observations_output.resolve(),
        history_output=args.history_output.resolve(),
        parents=parents,
        device_name=args.device,
        queries_per_fragment=args.queries_per_fragment,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EventHistorySources",
    "QueryPlan",
    "causal_history_mask",
    "compact_referenced_observations",
    "main",
    "materialize_event_history",
    "merge_fresh_pair_phase",
]
