"""Freeze the label-blind TRAIN40 RGB timeline and P/H/V role datasets."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.rgb_port.contracts import (
    LOOKBACK_US,
    MAX_OBSERVATIONS,
    RGB_SIZE,
    ROLE_COUNTS,
    SPLIT_SALT,
    RGBFrame,
    assign_roles,
    assignment_sha256,
    validate_role_sets,
)
from e_jepa_ttc.rgb_port.data import TarFrameReader, as_list, decode_query
from e_jepa_ttc.rgb_port.history import (
    observation_triplets,
    producer_frames,
    select_frames,
    sensor_timestamp_us,
)
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc

LABEL_FREE_COLUMNS = (
    "sequence_id",
    "sample_token",
    "track_id",
    "public_track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "rgb_shard_paths",
    "rgb_member_paths",
    "boxes_xyxy",
)
TARGET_COLUMNS = ("sample_token", "ttc")
TERMINAL_SCHEMA = "rgb_port_train40_inputs_v1"
_PILOT_READER: TarFrameReader | None = None
_PILOT_ROOT: Path | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _replace(source: Path, target: Path) -> None:
    delay = 0.02
    for attempt in range(10):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name, suffix=".pending", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace(pending, path)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f".{os.getpid()}.pending")
    try:
        frame.to_parquet(pending, index=False)
        with pending.open("r+b") as stream:
            os.fsync(stream.fileno())
        _replace(pending, path)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def _pilot_init(eap_root: str) -> None:
    global _PILOT_READER, _PILOT_ROOT
    _PILOT_ROOT = Path(eap_root)
    _PILOT_READER = TarFrameReader(_PILOT_ROOT, max_open=4)


def _pilot_task(row: dict[str, Any], destination: str) -> dict[str, Any]:
    if _PILOT_READER is None or _PILOT_ROOT is None:
        raise RuntimeError("pilot worker was not initialized")
    started = time.perf_counter()
    decoded = decode_query(row, eap_root=_PILOT_ROOT, reader=_PILOT_READER)
    token = str(row["sample_token"])
    path = Path(destination) / f"{hashlib.sha256(token.encode()).hexdigest()}.npz"
    pending = path.with_name(path.name + f".{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.savez_compressed(
            stream,
            rgb_uint8=decoded["rgb_uint8"],
            frame_times_us=decoded["frame_times_us"],
            delta_t_s=decoded["delta_t_s"],
            boxes_in_crop_xyxy=decoded["boxes_in_crop_xyxy"],
            roi_xyxy=decoded["roi_xyxy"],
        )
        stream.flush()
        os.fsync(stream.fileno())
    _replace(pending, path)
    return {
        "sample_token": token,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "elapsed_s": time.perf_counter() - started,
        "rgb_sha256": hashlib.sha256(decoded["rgb_uint8"].tobytes()).hexdigest(),
        "status": "COMPLETE",
    }


def _resource_snapshot(destination: Path) -> tuple[bool, dict[str, int]]:
    import psutil

    process = psutil.Process()
    tree = [process, *process.children(recursive=True)]
    rss = sum(item.memory_info().rss for item in tree if item.is_running())
    available = int(psutil.virtual_memory().available)
    disk = int(__import__("shutil").disk_usage(destination).free)
    values = {"tree_rss_bytes": rss, "host_available_bytes": available, "disk_free_bytes": disk}
    return available >= 2 * 1024**3 and rss <= 23_000_000_000 and disk >= 10_000_000_000, values


def run_pilot(
    rows_path: Path,
    eap_root: Path,
    destination: Path,
    count: int,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Decode a bounded role pilot with persistent workers and prefetch eight."""
    if not 1 <= count <= 512:
        raise ValueError("pilot query count must be between 1 and 512")
    if not 1 <= workers <= 4:
        raise ValueError("pilot workers must be between 1 and 4")
    frame = pd.read_parquet(rows_path)
    t2 = frame[frame.producer_frame_count == 2]
    t3 = frame[frame.producer_frame_count == 3]
    half = count // 2
    chosen = pd.concat((t2.head(min(half, len(t2))), t3.head(count - min(half, len(t2)))))
    if len(chosen) < count:
        remainder = frame[~frame.sample_token.isin(chosen.sample_token)].head(count - len(chosen))
        chosen = pd.concat((chosen, remainder))
    destination.mkdir(parents=True, exist_ok=True)
    allowed, before = _resource_snapshot(destination)
    if not allowed:
        raise RuntimeError(f"resource guard rejected RGB pilot: {before}")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    iterator = iter([cast(dict[str, Any], row.to_dict()) for _, row in chosen.iterrows()])
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_pilot_init, initargs=(str(eap_root.resolve()),)
    ) as pool:
        pending: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
        exhausted = False
        while pending or not exhausted:
            while len(pending) < 8 and not exhausted:
                allowed, resources = _resource_snapshot(destination)
                if not allowed:
                    raise RuntimeError(
                        f"resource guard paused before pilot submission: {resources}"
                    )
                try:
                    row = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = pool.submit(_pilot_task, row, str(destination))
                pending[future] = str(row["sample_token"])
            if pending:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    token = pending.pop(future)
                    try:
                        results.append(future.result())
                    except Exception as error:
                        failures.append({"sample_token": token, "error": repr(error)})
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": "rgb_port_cache_pilot_v1",
        "status": "COMPLETE" if not failures else "FAILED",
        "requested": count,
        "completed": len(results),
        "failures": failures,
        "workers": workers,
        "bounded_prefetch": 8,
        "elapsed_s": elapsed,
        "queries_per_second": len(results) / elapsed,
        "compressed_bytes": sum(item["bytes"] for item in results),
        "estimated_full_compressed_bytes": (
            int(sum(item["bytes"] for item in results) / len(results) * 88_744) if results else None
        ),
        "resources_before": before,
        "results": results,
    }
    atomic_json(destination / "PILOT_RECEIPT.json", receipt)
    if failures:
        raise RuntimeError(f"RGB pilot had {len(failures)} failures")
    return receipt


def _source_frame(row: Mapping[str, Any], index: int) -> RGBFrame:
    track = str(row["track_id"])
    member = str(as_list(row["rgb_member_paths"])[index])
    timestamp = int(as_list(row["frame_timestamps_us"])[index])
    box = tuple(float(value) for value in as_list(row["boxes_xyxy"])[index])
    return RGBFrame(
        frame_id=f"{track}|{member}",
        timestamp_us=timestamp,
        available_us=timestamp,
        shard_path=str(as_list(row["rgb_shard_paths"])[index]),
        member_path=member,
        box_xyxy=box,  # type: ignore[arg-type]
    )


def build_label_blind_index(metadata: pd.DataFrame, roles: Mapping[str, str]) -> pd.DataFrame:
    """Build causal RGB histories without reading a TTC value."""
    if any(name not in metadata.columns for name in LABEL_FREE_COLUMNS):
        raise ValueError("TRAIN40 metadata lacks required label-free columns")
    rows: list[dict[str, Any]] = []
    ordered = metadata.sort_values(["track_id", "timestamp_us", "sample_token"], kind="stable")
    timelines: dict[str, dict[str, RGBFrame]] = defaultdict(dict)
    for ordinal, record in enumerate(ordered.to_dict(orient="records")):
        track = str(record["track_id"])
        for index in range(len(as_list(record["frame_timestamps_us"]))):
            frame = _source_frame(record, index)
            previous = timelines[track].get(frame.frame_id)
            if previous is not None and previous != frame:
                raise ValueError(f"frame identity changed within track: {frame.frame_id}")
            timelines[track][frame.frame_id] = frame
        query_us = int(record["timestamp_us"])
        history = list(timelines[track].values())
        causal_history = select_frames(history, cutoff_us=query_us)
        selected = producer_frames(history, cutoff_us=query_us)
        observations = observation_triplets(history, cutoff_us=query_us)
        current = selected[-1] if selected else None
        roi = (
            common_square_from_boxes(
                [frame.box_xyxy for frame in selected[-2:]], (0, 1), margin_fraction=0.25
            )
            if len(selected) >= 2
            else (0.0, 0.0, 0.0, 0.0)
        )
        row = {
            "ordinal": ordinal,
            "source_ordinal": int(record.get("source_ordinal", ordinal)),
            "sample_token": str(record["sample_token"]),
            "sequence_id": str(record["sequence_id"]),
            "group_id": str(record["sequence_id"]),
            "track_id": track,
            "public_track_id": str(record["public_track_id"]),
            "role": str(roles[str(record["sequence_id"])]),
            "query_time_us": query_us,
            "cutoff_us": query_us,
            "rgb_available": len(selected) >= 2,
            "frame_ids": [frame.frame_id for frame in causal_history[-(MAX_OBSERVATIONS + 2) :]],
            "producer_frame_ids": [frame.frame_id for frame in selected],
            "producer_times_us": [frame.timestamp_us for frame in selected],
            "producer_available_us": [frame.available_us for frame in selected],
            "producer_sensor_times_us": [
                sensor_timestamp_us(frame.member_path) for frame in selected
            ],
            "producer_shards": [frame.shard_path for frame in selected],
            "producer_members": [frame.member_path for frame in selected],
            "producer_boxes_xyxy": [list(frame.box_xyxy) for frame in selected],
            "distinct_frames": len(causal_history[-(MAX_OBSERVATIONS + 2) :]),
            "producer_frame_count": len(selected),
            "distinct_observations": len(observations),
            "observation_ids": [
                hashlib.sha256(f"{track}\0{triplet[-1].frame_id}".encode()).hexdigest()
                for triplet in observations
            ],
            "observation_frame_ids": [
                [frame.frame_id for frame in triplet] for triplet in observations
            ],
            "observation_times_us": [
                [frame.timestamp_us for frame in triplet] for triplet in observations
            ],
            "observation_available_us": [
                [frame.available_us for frame in triplet] for triplet in observations
            ],
            "observation_sensor_times_us": [
                [sensor_timestamp_us(frame.member_path) for frame in triplet]
                for triplet in observations
            ],
            "observation_shards": [
                [frame.shard_path for frame in triplet] for triplet in observations
            ],
            "observation_members": [
                [frame.member_path for frame in triplet] for triplet in observations
            ],
            "observation_boxes_xyxy": [
                [list(frame.box_xyxy) for frame in triplet] for triplet in observations
            ],
            "observation_rois_xyxy": [
                list(
                    common_square_from_boxes(
                        [frame.box_xyxy for frame in triplet[-2:]],
                        (0, 1),
                        margin_fraction=0.25,
                    )
                )
                for triplet in observations
            ],
            "observation_anchor_us": [triplet[-1].timestamp_us for triplet in observations],
            "observation_query_age_us": [
                query_us - triplet[-1].timestamp_us for triplet in observations
            ],
            "actual_span_us": (
                selected[-1].timestamp_us - selected[0].timestamp_us if selected else 0
            ),
            "rgb_anchor_us": (current.timestamp_us if current is not None else -1),
            "query_delay_us": (query_us - current.timestamp_us if current is not None else -1),
            "roi_xyxy": list(roi),
            "calibration_id": "eap_public_train_rgb_bbox_space_v1",
            "foreground_source": "public_train_bbox_geometry",
            "dino_teacher_available": False,
        }
        rows.append(row)
    indexed = pd.DataFrame.from_records(rows).sort_values("ordinal", kind="stable")
    if indexed.sample_token.duplicated().any():
        raise ValueError("duplicate TRAIN40 sample tokens")
    if (indexed.query_delay_us < 0).any() or (indexed.actual_span_us > LOOKBACK_US).any():
        raise ValueError("causal RGB timing contract failed")
    if (indexed.distinct_observations > MAX_OBSERVATIONS).any():
        raise ValueError("RGB observation capacity exceeded")
    return indexed


def _role_payload(indexed: pd.DataFrame, role: str) -> pd.DataFrame:
    result = cast(pd.DataFrame, indexed[indexed.role == role].copy())
    available = cast(pd.DataFrame, result[result.rgb_available].copy())
    group_sizes = available.groupby("group_id").size().to_dict()
    group_count = len(group_sizes)
    available["mass"] = [1.0 / (group_count * group_sizes[group]) for group in available.group_id]
    return cast(pd.DataFrame, available.reset_index(drop=True))


def prepare(
    source_path: Path,
    output_root: Path,
    metadata_root: Path,
    eap_root: Path,
) -> dict[str, Any]:
    """Create immutable role/index artifacts from authorized TRAIN40 only."""
    for path in (source_path, eap_root):
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = pd.read_parquet(source_path, columns=list(LABEL_FREE_COLUMNS)).reset_index(
        names="source_ordinal"
    )
    groups = sorted(metadata.sequence_id.astype(str).unique().tolist())
    roles = assign_roles(groups)
    role_groups = {
        role: sorted(group for group, assigned in roles.items() if assigned == role)
        for role in ROLE_COUNTS
    }
    validate_role_sets(role_groups)
    if {role: len(values) for role, values in role_groups.items()} != ROLE_COUNTS:
        raise AssertionError("frozen role cardinality changed")
    assignment_hash = assignment_sha256(roles)  # label-free split is now frozen
    indexed = build_label_blind_index(metadata, roles)
    source_hash = file_sha256(source_path)
    split_path = output_root / "SPLIT_MANIFEST.json"
    split = {
        "schema": "rgb_port_split_manifest_v1",
        "status": "COMPLETE",
        "salt": SPLIT_SALT,
        "source": str(source_path.resolve()),
        "source_sha256": source_hash,
        "label_blind_index": True,
        "grouping_level": "sequence_id",
        "independence_limitation": (
            "provisional_sequence_held_out; acquisition metadata unavailable "
            "and TRAIN40 was historically exposed"
        ),
        "roles": {
            role: {
                "group_ids": role_groups[role],
                "group_count": len(role_groups[role]),
                "query_count": int((indexed.role == role).sum()),
            }
            for role in ROLE_COUNTS
        },
        "assignment_sha256": assignment_hash,
        "intersections": {"P_H": 0, "P_V": 0, "H_V": 0},
    }
    atomic_json(split_path, split)

    # Targets are loaded only after the label-blind split and frame index exist in memory.
    targets = pd.read_parquet(source_path, columns=list(TARGET_COLUMNS))
    target_map = targets.set_index("sample_token").ttc
    if set(indexed.sample_token) != set(target_map.index.astype(str)):
        raise ValueError("target rows do not align with the frozen label-blind index")
    indexed["target_ttc"] = indexed.sample_token.map(target_map).astype(np.float64)
    indexed["target_phase"] = expert_phase_from_ttc(indexed.target_ttc.to_numpy(np.float64))
    if not np.isfinite(indexed.target_phase).all():
        raise ValueError("nonfinite TRAIN40 target phase")

    metadata_root.mkdir(parents=True, exist_ok=True)
    query_path = metadata_root / "QUERY_INDEX.parquet"
    frame_path = metadata_root / "FRAME_INDEX.parquet"
    query_columns = [
        column for column in indexed.columns if column not in {"target_ttc", "target_phase", "mass"}
    ]
    atomic_parquet(query_path, cast(pd.DataFrame, indexed[query_columns]))
    atomic_parquet(
        frame_path,
        cast(
            pd.DataFrame,
            indexed[
                [
                    "sample_token",
                    "track_id",
                    "query_time_us",
                    "cutoff_us",
                    "frame_ids",
                    "producer_frame_ids",
                    "producer_times_us",
                    "producer_available_us",
                    "producer_sensor_times_us",
                    "producer_shards",
                    "producer_members",
                    "producer_boxes_xyxy",
                    "distinct_frames",
                    "producer_frame_count",
                    "distinct_observations",
                    "observation_ids",
                    "observation_frame_ids",
                    "observation_times_us",
                    "observation_available_us",
                    "observation_sensor_times_us",
                    "observation_shards",
                    "observation_members",
                    "observation_boxes_xyxy",
                    "observation_rois_xyxy",
                    "observation_anchor_us",
                    "observation_query_age_us",
                    "actual_span_us",
                    "rgb_anchor_us",
                    "query_delay_us",
                    "roi_xyxy",
                    "calibration_id",
                ]
            ],
        ),
    )
    role_manifests: dict[str, str] = {}
    role_stats: dict[str, Any] = {}
    for role in ROLE_COUNTS:
        role_frame = _role_payload(indexed, role)
        rows_path = metadata_root / f"{role}_ROWS.parquet"
        atomic_parquet(rows_path, role_frame)
        rows_hash = file_sha256(rows_path)
        manifest_path = output_root / f"{role}_MANIFEST.json"
        manifest = {
            "schema": "rgb_port_role_manifest_v1",
            "status": "COMPLETE",
            "role": role,
            "role_scope": {
                "P": "producer_and_pair_fit",
                "H": "normalizer_and_head_fit",
                "V": "selection_and_evaluation_only",
            }[role],
            "eap_root": str(eap_root.resolve()),
            "rows_path": str(rows_path.resolve()),
            "rows_sha256": rows_hash,
            "source_sha256": source_hash,
            "split_assignment_sha256": assignment_hash,
            "group_ids": role_groups[role],
            "group_count": len(role_groups[role]),
            "query_count_total": int((indexed.role == role).sum()),
            "population_size": len(role_frame),
            "t2_count": int((role_frame.producer_frame_count == 2).sum()),
            "t3_count": int((role_frame.producer_frame_count == 3).sum()),
            "label_blind_index": True,
            "targets_attached_after_freeze": True,
            "dino_teacher_binding": (
                "P_DINO_MANIFEST.json_required_for_fit" if role == "P" else "NOT_APPLICABLE"
            ),
        }
        manifest["identity_sha256"] = _json_sha(manifest)
        atomic_json(manifest_path, manifest)
        role_manifests[role] = str(manifest_path.resolve())
        role_stats[role] = {
            key: manifest[key]
            for key in ("query_count_total", "population_size", "t2_count", "t3_count")
        }

    frame_manifest = {
        "schema": "rgb_port_frame_availability_v1",
        "status": "COMPLETE",
        "label_blind_index": True,
        "query_index_path": str(query_path.resolve()),
        "query_index_sha256": file_sha256(query_path),
        "frame_index_path": str(frame_path.resolve()),
        "frame_index_sha256": file_sha256(frame_path),
        "max_context": MAX_OBSERVATIONS,
        "lookback_us": LOOKBACK_US,
        "producer_context": "T3_when_available_else_genuine_T2_cold_start",
        "no_interpolation_or_frame_duplication": True,
        "capture_availability_query_clocks_separate": True,
        "calibration": {
            "id": "eap_public_train_rgb_bbox_space_v1",
            "dataset": "eAP public TRAIN",
            "camera_space": "public RGB frame and public RGB bbox coordinates",
            "roi": "single current-query square applied unchanged to every observation frame",
            "fcwd_transfer": False,
        },
        "roles": role_stats,
    }
    atomic_json(output_root / "FRAME_AVAILABILITY.json", frame_manifest)
    input_contracts = {
        "schema": "rgb_port_input_contracts_v1",
        "status": "COMPLETE",
        "dataset_schema": TERMINAL_SCHEMA,
        "rgb": {
            "dtype": "float32",
            "range": [0.0, 1.0],
            "shape": ["B", "T=2|3", 3, RGB_SIZE, RGB_SIZE],
            "normalization": "none",
        },
        "teacher_rgb_uint8": {
            "dtype": "uint8",
            "shape": ["B", "T=2|3", 3, RGB_SIZE, RGB_SIZE],
            "route": "frozen_DINO_teacher_only",
        },
        "frame_times_us": {"dtype": "int64", "shape": ["B", "T"]},
        "annotation_frame_times_us": {
            "dtype": "int64",
            "shape": ["B", "T"],
            "use": "query_history_and_label_anchor_mapping",
        },
        "sensor_frame_times_us": {
            "dtype": "int64",
            "shape": ["B", "T"],
            "use": "real_RGB_same_sensor_motion_clock",
        },
        "delta_t_s": {
            "dtype": "float32",
            "shape": ["B", "T-1"],
            "derivation": "same_sensor_integer_microsecond_difference_then_float_conversion",
        },
        "target_ttc": {
            "manifest_dtype": "float64",
            "training_tensor_dtype": "float32",
            "shape": ["B"],
            "model_input": False,
            "anchor": "query_time_us",
        },
        "target_phase": {
            "manifest_dtype": "float64",
            "training_tensor_dtype": "float32",
            "shape": ["B"],
            "model_input": False,
        },
        "foreground_mask": {
            "dtype": "bool",
            "shape": ["B", "T", 1, RGB_SIZE, RGB_SIZE],
            "source": "public_train_bbox_geometry",
        },
        "normalization_roles": {"producer": "P", "heads": "H", "evaluation": "V_never_fit"},
        "role_manifests": role_manifests,
    }
    atomic_json(output_root / "INPUT_CONTRACTS.json", input_contracts)
    estimated_uint8 = int(
        sum(value["population_size"] for value in role_stats.values()) * 3 * 3 * RGB_SIZE * RGB_SIZE
    )
    receipt = {
        "schema": "rgb_port_prepare_receipt_v1",
        "status": "COMPLETE",
        "source_sha256": source_hash,
        "split_assignment_sha256": assignment_hash,
        "role_manifests": role_manifests,
        "role_stats": role_stats,
        "rgb_cache_written": False,
        "estimated_uncompressed_three_frame_uint8_bytes": estimated_uint8,
        "pilot_required_before_full_cache": True,
    }
    atomic_json(output_root / "PREPARE_RECEIPT.json", receipt)
    return receipt


def admit_historical_dino(
    *,
    role_manifest_path: Path,
    teacher_manifest_path: Path,
    teacher_index_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Admit the frozen TRAIN40 generic teacher only after exact P frame/ROI parity."""
    role_manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    if role_manifest.get("status") != "COMPLETE" or role_manifest.get("role") != "P":
        raise ValueError("DINO admission requires the COMPLETE P role manifest")
    rows = pd.read_parquet(role_manifest["rows_path"])
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    if teacher_manifest.get("status") != "COMPLETE_VERIFIED":
        raise ValueError("historical generic teacher manifest is not COMPLETE_VERIFIED")
    if teacher_manifest.get("rows_sha256") != role_manifest.get("source_sha256"):
        raise ValueError("historical teacher and RGB-PORT source rows differ")
    if file_sha256(teacher_index_path) != teacher_manifest.get("index_sha256"):
        raise ValueError("historical teacher index hash mismatch")
    with np.load(teacher_index_path, allow_pickle=False) as stored:
        tokens = stored["tokens"]
        members = stored["rgb_members"]
        squares = stored["square_xyxy"]
        for row in rows.to_dict(orient="records"):
            source_ordinal = int(row["source_ordinal"])
            if str(tokens[source_ordinal]) != str(row["sample_token"]):
                raise ValueError("P token differs from historical generic teacher")
            if list(map(str, members[source_ordinal])) != list(
                map(str, as_list(row["producer_members"])[-2:])
            ):
                raise ValueError("P endpoint frame IDs differ from historical generic teacher")
            if not np.array_equal(
                np.asarray(squares[source_ordinal], dtype=np.float64),
                np.asarray(as_list(row["roi_xyxy"]), dtype=np.float64),
            ):
                raise ValueError("P shared ROI differs from historical generic teacher")
    directory = Path(teacher_manifest["directory"]).resolve(strict=True)
    shards: list[dict[str, Any]] = []
    for item in teacher_manifest["ordered_shards"]:
        path = directory / item["name"]
        observed = file_sha256(path)
        if observed != item["sha256"]:
            raise ValueError(f"historical generic teacher shard changed: {path.name}")
        shards.append(
            {
                "path": str(path.resolve()),
                "sha256": observed,
                "start": int(item["start"]),
                "stop": int(item["stop"]),
            }
        )
    freeze_path = teacher_manifest_path.with_name("TEACHER_FREEZE.json")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    alignment_payload = [
        [
            int(row["source_ordinal"]),
            str(row["sample_token"]),
            *list(map(str, as_list(row["producer_members"])[-2:])),
            *list(map(int, as_list(row["roi_xyxy"]))),
        ]
        for row in cast(list[dict[str, Any]], rows.to_dict(orient="records"))
    ]
    result = {
        "schema": "rgb_port_dino_sidecar_manifest_v1",
        "status": "COMPLETE",
        "role": "P",
        "role_manifest_path": str(role_manifest_path.resolve()),
        "role_manifest_sha256": file_sha256(role_manifest_path),
        "population_size": len(rows),
        "input_storage": "raw_uint8_rgb_no_double_divide",
        "teacher_input_endpoints": "last_two_real_producer_frames",
        "target_anchor": "RGB_frame_endpoints_with_query_delay_explicit",
        "frame_roi_alignment_sha256": _json_sha(alignment_payload),
        "teacher_manifest_path": str(teacher_manifest_path.resolve()),
        "teacher_manifest_sha256": file_sha256(teacher_manifest_path),
        "teacher_index_path": str(teacher_index_path.resolve()),
        "teacher_index_sha256": file_sha256(teacher_index_path),
        "teacher_freeze_path": str(freeze_path.resolve()),
        "teacher_freeze_sha256": file_sha256(freeze_path),
        "teacher_weights_sha256": freeze["ordered_tensor_weights_sha256"],
        "teacher_model_revision": freeze["model_revision"],
        "relation_shape_per_query": [2, 6, 32, 32],
        "relation_storage_dtype": "float16",
        "shards": shards,
        "all_shards_sha256_verified": True,
        "ttc_targets_read_by_teacher": False,
    }
    atomic_json(output_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--metadata-root", type=Path, required=True)
    result.add_argument("--eap-root", type=Path, required=True)
    result.add_argument("--pilot-queries", type=int, default=0)
    result.add_argument("--pilot-cache-root", type=Path)
    result.add_argument("--teacher-manifest", type=Path)
    result.add_argument("--teacher-index", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    receipt = prepare(args.source, args.output, args.metadata_root, args.eap_root)
    if (args.teacher_manifest is None) != (args.teacher_index is None):
        raise ValueError("teacher manifest and index must be supplied together")
    if args.teacher_manifest is not None and args.teacher_index is not None:
        receipt = {
            **receipt,
            "dino_sidecar": admit_historical_dino(
                role_manifest_path=args.output / "P_MANIFEST.json",
                teacher_manifest_path=args.teacher_manifest,
                teacher_index_path=args.teacher_index,
                output_path=args.output / "P_DINO_MANIFEST.json",
            ),
        }
    if args.pilot_queries:
        if args.pilot_cache_root is None:
            raise ValueError("--pilot-cache-root is required with --pilot-queries")
        receipt = {
            **receipt,
            "pilot": run_pilot(
                args.metadata_root / "P_ROWS.parquet",
                args.eap_root,
                args.pilot_cache_root,
                args.pilot_queries,
            ),
        }
    atomic_json(args.output / "PREPARE_RECEIPT.json", receipt)
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "status": receipt["status"],
                "role_stats": receipt["role_stats"],
                "split_assignment_sha256": receipt["split_assignment_sha256"],
                "dino_sidecar_status": (
                    receipt.get("dino_sidecar", {}).get("status")
                    if isinstance(receipt.get("dino_sidecar"), dict)
                    else None
                ),
                "pilot_status": (
                    receipt.get("pilot", {}).get("status")
                    if isinstance(receipt.get("pilot"), dict)
                    else None
                ),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
