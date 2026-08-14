"""Train-only, auditable temporal cache for Scientific Recovery V8.

This module intentionally does not extend the historical V4 cache.  V8 stores
only the event-derived temporal representation required by its model plus the
separate supervision/provenance fields needed by the grouped-development
protocol.  In particular, the cache never materializes public validation,
private-test, EvTTC-test, or CodaBench examples.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from csv import DictReader
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE, EAPEventReader
from e_jepa_ttc.data.event_v4_geometry import (
    common_square_from_boxes,
    shifted_precontext_window,
)
from e_jepa_ttc.data.garlttc_eap import (
    GARLTTC_JOIN_KEYS,
    load_garlttc_train_index,
    normalize_boxes_xyxy,
    normalize_event_windows_us,
    resolve_eap_events_path,
)
from e_jepa_ttc.data.garlttc_lhr_cache import (
    _official_ttc_at_endpoint,
    select_temporal_indices,
)
from e_jepa_ttc.data.garlttc_sampling import signed_ttc_bucket
from e_jepa_ttc.data.scientific_recovery_v8 import (
    CausalExponentialStateRepresentation,
    GarlTimeVolumeRepresentation,
    ScientificRecoveryV8Batch,
)
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.evaluation.garl_ttc_protocol import PAPER_MID_WEIGHTS
from e_jepa_ttc.utils.io import read_structured

V8_CACHE_SCHEMA = "scientific_recovery_v8_temporal_cache_v1"
V8_CACHE_FORMAT = "torch_sharded_list_v1"
_FORBIDDEN_PATH_TOKENS = ("validation", "test_inputs", "private", "codabench", "evttc")
_REPRESENTATIONS = {"timevol20", "exp6"}


@dataclass(frozen=True)
class ScientificRecoveryV8CacheConfig:
    """Immutable V8 temporal-cache controls.

    ``selection_metadata_path`` must identify the already frozen 8,192 public
    train rows.  It is deliberately a metadata input rather than a cache input:
    representations are always rebuilt from raw public-train event HDF5 files.
    """

    representation: Literal["timevol20", "exp6"]
    steps: int = 3
    roi_size: int = 128
    timevol_planes: int = 20
    timevol_window_ms: float = 100.0
    exp6_alphas: tuple[float, ...] = (0.1, 0.05, 0.025, 0.0125, 0.0075, 0.0035)
    exp6_internal_dt_ms: float = 0.2
    target_delta_t_s: float = 0.1
    delta_t_tolerance_s: float = 0.025
    context_shift_s: float = 0.1
    roi_margin_fraction: float = 0.25
    shard_size: int = 128
    storage_dtype: Literal["float16", "float32"] = "float16"
    expected_rows: int | None = 8192
    require_protocol_identity: bool = True

    def __post_init__(self) -> None:
        if self.representation not in _REPRESENTATIONS:
            raise ValueError(f"representation must be one of {sorted(_REPRESENTATIONS)}")
        if self.steps not in {2, 3}:
            raise ValueError("V8 cache steps must be 2 or 3")
        if min(self.roi_size, self.timevol_planes, self.shard_size) <= 0:
            raise ValueError("roi_size, timevol_planes, and shard_size must be positive")
        if self.timevol_window_ms <= 0.0 or self.context_shift_s <= 0.0:
            raise ValueError("temporal windows must be positive")
        if self.target_delta_t_s <= 0.0 or self.delta_t_tolerance_s <= 0.0:
            raise ValueError("endpoint pairing controls must be positive")
        if self.roi_margin_fraction < 0.0:
            raise ValueError("roi_margin_fraction must be non-negative")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype must be float16 or float32")
        if self.expected_rows is not None and self.expected_rows <= 0:
            raise ValueError("expected_rows must be positive or None")

    @property
    def channels(self) -> int:
        """Return the model channel count for one endpoint."""

        return self.timevol_planes if self.representation == "timevol20" else len(self.exp6_alphas)


@dataclass(frozen=True)
class _PlannedRow:
    identity: tuple[str, ...]
    sequence_id: str
    sample_token: str
    track_id: str
    endpoint_us: tuple[int, ...]
    roi_xyxy: tuple[float, float, float, float]
    endpoint_boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    target_ttc: float
    target_text: str
    sample_weight: float
    sample_weight_text: str
    outer_fold: int
    events_path: Path


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = dict(value)
    sign_artifact(payload)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, list) or not all(isinstance(row, dict) for row in loaded):
        raise ValueError(f"V8 shard {path} is not a list of record mappings")
    return loaded


def _identity(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in GARLTTC_JOIN_KEYS)


def _token_hash(tokens: Iterable[str]) -> str:
    return _canonical_hash(sorted(str(value) for value in tokens))


def _assert_train_only_path(path: Path, *, label: str) -> None:
    lowered = path.as_posix().lower()
    if any(token in lowered for token in _FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"{label} may not reference a sealed/non-train source: {path}")


def _protocol_rows(
    protocol: Mapping[str, Any],
) -> tuple[int | None, str | None, list[Mapping[str, Any]]]:
    contract = protocol.get("sample_contract", protocol)
    if not isinstance(contract, Mapping):
        raise ValueError("protocol has no sample contract mapping")
    rows = contract.get("rows", contract.get("sample_count"))
    expected_rows = int(rows) if rows is not None else None
    token_hash = contract.get("sorted_sample_tokens_sha256")
    folds = contract.get("fold_definitions", contract.get("folds", []))
    if isinstance(folds, int):
        folds = protocol.get("folds", [])
    if not isinstance(folds, list):
        raise ValueError("protocol folds must be a list")
    return expected_rows, str(token_hash) if token_hash is not None else None, folds


def _canonical_records(records: Sequence[Mapping[str, str]]) -> str:
    """Match the newline-delimited canonical hash contract frozen by V8."""

    payload = b"".join(
        json.dumps(dict(record), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
        for record in records
    )
    return hashlib.sha256(payload).hexdigest()


def _protocol_source_path(protocol: Mapping[str, Any], key: str) -> Path:
    sources = protocol.get("sources")
    if not isinstance(sources, Mapping) or not isinstance(sources.get(key), Mapping):
        raise ValueError(f"protocol does not declare sources.{key}")
    value = sources[key].get("path")
    if not isinstance(value, str):
        raise ValueError(f"protocol sources.{key}.path must be a string")
    path = (Path(__file__).resolve().parents[3] / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _selection_from_paired_oof(protocol: Mapping[str, Any]) -> list[dict[str, str]]:
    """Derive the exact V8 universe from the two signed train-only OOF files."""

    a5_path = _protocol_source_path(protocol, "a5_oof_predictions")
    garl_path = _protocol_source_path(protocol, "garl_oof_predictions")
    _assert_train_only_path(a5_path, label="A5 OOF source")
    _assert_train_only_path(garl_path, label="Garl OOF source")
    with a5_path.open(newline="", encoding="utf-8") as handle:
        a5 = {row["sample_token"]: row for row in DictReader(handle)}
    with garl_path.open(newline="", encoding="utf-8") as handle:
        garl = {row["sample_token"]: row for row in DictReader(handle)}
    required = {"sample_token", "sequence_id", "track_id", "target_ttc_s", "fold"}
    if not a5 or set(a5) != set(garl) or not required.issubset(next(iter(a5.values()))):
        raise ValueError("paired V7 OOF sources do not satisfy the frozen V8 row contract")
    output: list[dict[str, str]] = []
    for token in sorted(a5):
        left, right = a5[token], garl[token]
        if any(left[field] != right[field] for field in ("sequence_id", "track_id", "fold")):
            raise ValueError(f"paired OOF identity mismatch for {token}")
        target_delta = abs(Decimal(left["target_ttc_s"]) - Decimal(right["target_ttc_s"]))
        if target_delta > Decimal("0.000001"):
            raise ValueError(f"paired OOF target mismatch for {token}")
        output.append(
            {
                "sequence_id": left["sequence_id"],
                "sample_token": token,
                "track_id": left["track_id"],
                "target_ttc": left["target_ttc_s"],
                "outer_fold": left["fold"],
            }
        )
    contract = protocol.get("sample_contract")
    expected_hash = (
        contract.get("ordered_token_ids_sha256") if isinstance(contract, Mapping) else None
    )
    if isinstance(expected_hash, str):
        actual_hash = _canonical_records([{"token_id": row["sample_token"]} for row in output])
        if actual_hash != expected_hash:
            raise ValueError("paired OOF token order does not match frozen protocol identity")
    return output


def _outer_fold_by_sequence(protocol: Mapping[str, Any]) -> dict[str, int]:
    _, _, definitions = _protocol_rows(protocol)
    mapping: dict[str, int] = {}
    for definition in definitions:
        if not isinstance(definition, Mapping):
            raise ValueError("protocol fold definition must be a mapping")
        fold = int(definition["fold"])
        sequences = definition.get("dev_sequence_ids", definition.get("outer_dev_sequence_ids", []))
        if not isinstance(sequences, list):
            raise ValueError("protocol dev_sequence_ids must be a list")
        for sequence in sequences:
            sequence_id = str(sequence)
            if sequence_id in mapping:
                raise ValueError(f"sequence {sequence_id} is dev in multiple outer folds")
            mapping[sequence_id] = fold
    return mapping


def _repository_provenance() -> dict[str, str | None]:
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    diff = subprocess.run(["git", "diff", "--binary"], cwd=root, capture_output=True, check=False)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    untracked: list[dict[str, str]] = []
    for line in status.stdout.splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        candidate = root / relative
        if candidate.is_file() and candidate.suffix in {".py", ".yaml", ".json"}:
            untracked.append(
                {"path": relative.replace("\\", "/"), "sha256": _file_sha256(candidate)}
            )
    untracked.sort(key=lambda item: item["path"])
    return {
        "code_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        "relevant_untracked_sha256": _canonical_hash(untracked),
    }


def _raw_event_batch(
    raw: Mapping[str, np.ndarray], *, sequence_id: str, start_us: int, end_us: int
) -> EventBatch:
    """Construct a validated, polarity-normalized raw full-frame event batch."""

    polarity = np.where(np.asarray(raw["p"]) > 0, 1, -1).astype(np.int8, copy=False)
    return EventBatch(
        x=np.asarray(raw["x"], dtype=np.int32),
        y=np.asarray(raw["y"], dtype=np.int32),
        t_us=np.asarray(raw["t"], dtype=np.int64),
        polarity=polarity,
        width=EAP_IMAGE_SIZE[0],
        height=EAP_IMAGE_SIZE[1],
        sequence_id=sequence_id,
        t_start_us=int(start_us),
        t_end_us=int(end_us),
    )


def _planned_rows(
    *,
    eap_root: Path,
    garlttc_root: Path,
    selection_metadata_path: Path | None,
    protocol_path: Path | None,
    config: ScientificRecoveryV8CacheConfig,
) -> tuple[list[_PlannedRow], dict[str, Any]]:
    """Join frozen row identities to official train metadata without opening validation."""

    _assert_train_only_path(garlttc_root / "data" / "train.parquet", label="GarlTTC data")
    _assert_train_only_path(eap_root / "data" / "train.parquet", label="eAP train metadata")
    protocol: Mapping[str, Any] | None = None
    expected_rows = config.expected_rows
    expected_token_hash: str | None = None
    fold_by_sequence: dict[str, int] = {}
    if protocol_path is not None:
        protocol_raw = read_structured(protocol_path)
        if not isinstance(protocol_raw, Mapping):
            raise ValueError("V8 protocol must be a mapping")
        protocol = protocol_raw
        protocol_rows, expected_token_hash, _ = _protocol_rows(protocol)
        if protocol_rows is not None:
            expected_rows = protocol_rows
        fold_by_sequence = _outer_fold_by_sequence(protocol)
    if selection_metadata_path is None:
        if protocol is None:
            raise ValueError("selection metadata is required without a frozen V8 protocol")
        selected_rows = _selection_from_paired_oof(protocol)
    else:
        selection = read_structured(selection_metadata_path)
        rows_value = (
            selection.get("rows", selection.get("selected_rows", selection.get("samples")))
            if isinstance(selection, Mapping)
            else selection
        )
        if not isinstance(rows_value, list) or not rows_value:
            raise ValueError("selection metadata must contain a non-empty rows/selected_rows list")
        if not all(isinstance(item, Mapping) for item in rows_value):
            raise ValueError("selection metadata rows must be mappings")
        selected_rows = [dict(item) for item in rows_value]
    selected_keys = [
        (
            str(row.get("sequence_id", "")),
            str(row.get("sample_token", "")),
            str(row.get("track_id", "")),
        )
        for row in selected_rows
    ]
    if any(not all(key) for key in selected_keys) or len(set(selected_keys)) != len(selected_keys):
        raise ValueError("selection rows require unique sequence_id/sample_token/track_id")
    if expected_rows is not None and len(selected_rows) != expected_rows:
        raise ValueError(f"selected rows={len(selected_rows)} but expected {expected_rows}")
    if expected_token_hash is not None:
        actual = _token_hash(key[1] for key in selected_keys)
        if actual != expected_token_hash:
            raise ValueError("selection sample-token hash does not match frozen protocol")
    elif config.require_protocol_identity:
        raise ValueError("require_protocol_identity=True requires --protocol")

    sequence_ids = sorted({key[0] for key in selected_keys})
    index = load_garlttc_train_index(garlttc_root, sequence_ids)
    merged = index.merged.copy()
    source_rows = merged.to_dict(orient="records")
    source = {
        (str(row["sequence_id"]), str(row["sample_token"]), str(row["track_id"])): row
        for row in source_rows
    }
    if len(source) != len(source_rows):
        raise ValueError("official train metadata has a duplicate V8 selection key")
    missing = [key for key in selected_keys if key not in source]
    if missing:
        raise ValueError(
            f"frozen selection has {len(missing)} rows missing from official train metadata"
        )

    plan: list[_PlannedRow] = []
    targets_from_contract: list[Decimal] = []
    for selected, selection_key in zip(selected_rows, selected_keys, strict=True):
        row = source[selection_key]
        identity = _identity(row)
        boxes = normalize_boxes_xyxy(row["boxes_xyxy"])
        windows = normalize_event_windows_us(row["event_windows_us"])
        timestamps = [int(value) for value in row["frame_timestamps_us"]]
        usable = min(len(boxes), len(windows), len(timestamps))
        if usable < 2:
            raise ValueError(f"{identity}: requires two aligned frames")
        boxes, windows, timestamps = boxes[:usable], windows[:usable], timestamps[:usable]
        first, second, _ = select_temporal_indices(
            timestamps,
            anchor_timestamp_us=int(row["timestamp_us"]),
            target_delta_t_s=config.target_delta_t_s,
            tolerance_s=config.delta_t_tolerance_s,
            context_delta_t_s=config.context_shift_s,
            context_tolerance_s=config.delta_t_tolerance_s,
        )
        context_window = shifted_precontext_window(windows[first], shift_s=config.context_shift_s)
        endpoints = (int(context_window[1]), int(windows[first][1]), int(windows[second][1]))
        if config.steps == 2:
            endpoints = endpoints[1:]
        if any(right <= left for left, right in zip(endpoints, endpoints[1:], strict=True)):
            raise ValueError(f"{identity}: temporal endpoints must strictly increase")
        target = float(_official_ttc_at_endpoint(row, second))
        target_text = str(selected.get("target_ttc", selected.get("target_ttc_s", target)))
        target_delta = abs(Decimal(target_text) - Decimal(str(target)))
        if not math.isfinite(target) or target_delta > Decimal("0.000001"):
            raise ValueError(f"{identity}: official target differs from frozen selection metadata")
        bucket = signed_ttc_bucket(target)
        if bucket not in PAPER_MID_WEIGHTS:
            raise ValueError(f"{identity}: TTC {target} lies outside signed MiD protocol")
        sequence = str(row["sequence_id"])
        if fold_by_sequence and sequence not in fold_by_sequence:
            raise ValueError(f"{identity}: sequence absent from frozen outer folds")
        default_fold = fold_by_sequence.get(sequence, -1)
        outer_fold = int(selected.get("outer_fold", selected.get("fold", default_fold)))
        if fold_by_sequence and outer_fold != fold_by_sequence[sequence]:
            raise ValueError(f"{identity}: selected outer fold differs from frozen sequence fold")
        targets_from_contract.append(Decimal(target_text))
        plan.append(
            _PlannedRow(
                identity=identity,
                sequence_id=sequence,
                sample_token=str(row["sample_token"]),
                track_id=str(row["track_id"]),
                endpoint_us=endpoints,
                roi_xyxy=common_square_from_boxes(
                    boxes, (first, second), margin_fraction=config.roi_margin_fraction
                ),
                endpoint_boxes_xyxy=tuple(
                    tuple(float(value) for value in boxes[index])
                    for index in ((first,) if config.steps == 2 else (first, first, second))
                ),
                target_ttc=target,
                target_text=target_text,
                sample_weight=0.0,
                sample_weight_text="",
                outer_fold=outer_fold,
                events_path=resolve_eap_events_path(eap_root, str(row["events_path"])),
            )
        )
    count_by_sequence_bucket: dict[tuple[str, str], int] = defaultdict(int)
    for item, target in zip(plan, targets_from_contract, strict=True):
        count_by_sequence_bucket[(item.sequence_id, signed_ttc_bucket(float(target)))] += 1
    weighted_plan: list[_PlannedRow] = []
    for item, target in zip(plan, targets_from_contract, strict=True):
        bucket = signed_ttc_bucket(float(target))
        coefficient = Decimal(str(PAPER_MID_WEIGHTS[bucket]))
        row_count = count_by_sequence_bucket[(item.sequence_id, bucket)]
        weight = coefficient / Decimal(9) / Decimal(row_count)
        weighted_plan.append(
            _PlannedRow(
                **{
                    **asdict(item),
                    "sample_weight": float(weight),
                    "sample_weight_text": str(weight),
                }
            )
        )
    plan = weighted_plan
    plan.sort(key=lambda item: (item.sequence_id, item.endpoint_us[-1], item.identity))
    contract_records = sorted(plan, key=lambda item: item.sample_token)
    frozen_contract = protocol.get("sample_contract", {}) if protocol is not None else {}
    expected_contract_hashes = {
        "row_identity_sha256": _canonical_records(
            [
                {
                    "sequence_id": item.sequence_id,
                    "token_id": item.sample_token,
                    "track_id": item.track_id,
                }
                for item in contract_records
            ]
        ),
        "target_sha256": _canonical_records(
            [
                {"target_ttc_s": item.target_text, "token_id": item.sample_token}
                for item in contract_records
            ]
        ),
        "mid_sample_weight_sha256": _canonical_records(
            [
                {"sample_weight": item.sample_weight_text, "token_id": item.sample_token}
                for item in contract_records
            ]
        ),
        "fold_assignment_sha256": _canonical_records(
            [
                {
                    "outer_fold": str(item.outer_fold),
                    "sequence_id": item.sequence_id,
                    "token_id": item.sample_token,
                }
                for item in contract_records
            ]
        ),
    }
    if protocol is not None:
        mismatches = {
            key: (frozen_contract.get(key), value)
            for key, value in expected_contract_hashes.items()
            if frozen_contract.get(key) != value
        }
        if mismatches:
            raise ValueError(f"frozen V8 row/target/weight/fold contract mismatch: {mismatches}")
    provenance = {
        "selection_metadata_path": (
            selection_metadata_path.as_posix() if selection_metadata_path else None
        ),
        "selection_metadata_sha256": (
            _file_sha256(selection_metadata_path) if selection_metadata_path else None
        ),
        "protocol_path": protocol_path.as_posix() if protocol_path is not None else None,
        "protocol_sha256": _file_sha256(protocol_path) if protocol_path is not None else None,
        "selected_rows": len(plan),
        **expected_contract_hashes,
        "sorted_sample_tokens_sha256": _token_hash(item.sample_token for item in plan),
        "garlttc_data_sha256": index.data_sha256,
        "garlttc_annotations_sha256": index.annotations_sha256,
        "garlttc_join_keys_sha256": index.join_keys_sha256,
        "outer_fold_by_sequence": dict(sorted(fold_by_sequence.items())),
        "uses_public_train_only": True,
        "uses_official_garl_ttc_labels": True,
        "uses_raw_public_train_events": True,
    }
    return plan, provenance


def _timevol_record(
    plan: _PlannedRow, reader: EAPEventReader, config: ScientificRecoveryV8CacheConfig
) -> dict[str, Any]:
    frontend = GarlTimeVolumeRepresentation(
        window_ms=config.timevol_window_ms,
        number_of_planes=config.timevol_planes,
        target_size=(config.roi_size, config.roi_size),
    )
    roi = torch.tensor(plan.roi_xyxy, dtype=torch.float32)
    outputs = []
    for endpoint in plan.endpoint_us:
        start = max(reader.t_start_us, endpoint - int(round(config.timevol_window_ms * 1_000.0)))
        raw = reader.read_window(start, endpoint + 1)
        events = _raw_event_batch(
            raw, sequence_id=plan.sequence_id, start_us=start, end_us=endpoint
        )
        outputs.append(frontend.encode(events, endpoint, roi))
    return _record_from_outputs(plan, outputs, config)


def _exp6_records_for_sequence(
    plans: Sequence[_PlannedRow], reader: EAPEventReader, config: ScientificRecoveryV8CacheConfig
) -> list[dict[str, Any]]:
    """Materialize an ordered sequence once, sharing one causal EXP6 state."""

    state = CausalExponentialStateRepresentation(
        alphas=config.exp6_alphas,
        internal_dt_ms=config.exp6_internal_dt_ms,
        target_size=(config.roi_size, config.roi_size),
    )
    requests: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_index, plan in enumerate(plans):
        for step, endpoint in enumerate(plan.endpoint_us):
            requests[int(endpoint)].append((row_index, step))
    output_slots: list[list[Any | None]] = [[None] * config.steps for _ in plans]
    watermark = reader.t_start_us
    for endpoint in sorted(requests):
        if endpoint < watermark:
            raise ValueError("EXP6 endpoint precedes stream watermark")
        raw = reader.read_window(watermark, endpoint + 1)
        packet = _raw_event_batch(
            raw,
            sequence_id=plans[0].sequence_id,
            start_us=reader.t_start_us,
            end_us=endpoint,
        )
        event_count = state.update(packet, endpoint)
        watermark = endpoint + 1
        for row_index, step in requests[endpoint]:
            plan = plans[row_index]
            output_slots[row_index][step] = state.snapshot(
                endpoint, torch.tensor(plan.roi_xyxy, dtype=torch.float32), event_count=event_count
            )
    records: list[dict[str, Any]] = []
    for plan, outputs in zip(plans, output_slots, strict=True):
        if any(value is None for value in outputs):
            raise RuntimeError("EXP6 sequence materialization left an endpoint unresolved")
        records.append(_record_from_outputs(plan, list(outputs), config))
    return records


def _record_from_outputs(
    plan: _PlannedRow, outputs: Sequence[Any], config: ScientificRecoveryV8CacheConfig
) -> dict[str, Any]:
    tensor = torch.stack([output.tensor for output in outputs]).contiguous()
    expected = (config.steps, config.channels, config.roi_size, config.roi_size)
    if tuple(tensor.shape) != expected or not torch.isfinite(tensor).all():
        raise ValueError(f"temporal frontend produced {tuple(tensor.shape)}, expected {expected}")
    storage = np.float16 if config.storage_dtype == "float16" else np.float32
    return {
        "representation": tensor.numpy().astype(storage),
        "endpoint_us": np.asarray(plan.endpoint_us, dtype=np.int64),
        "sample_token": plan.sample_token,
        "sequence_id": plan.sequence_id,
        "track_id": plan.track_id,
        "row_identity": list(plan.identity),
        "target_ttc": np.float32(plan.target_ttc),
        "sample_weight": np.float32(plan.sample_weight),
        "outer_fold": np.int64(plan.outer_fold),
        "common_roi_xyxy": np.asarray(plan.roi_xyxy, dtype=np.float32),
        "endpoint_boxes_xyxy": np.asarray(plan.endpoint_boxes_xyxy, dtype=np.float32),
        "visible_heights_px": np.asarray(
            [box[3] - box[1] for box in plan.endpoint_boxes_xyxy[-2:]], dtype=np.float32
        ),
        "representation_source": str(outputs[0].source),
        "endpoint_diagnostics": [dict(output.diagnostics) for output in outputs],
    }


def _write_records(
    *,
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    config: ScientificRecoveryV8CacheConfig,
    provenance: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    """Atomically persist prevalidated records and return a signed manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "build_state.json"
    manifest_path = output_dir / "manifest.json"
    identity_hash = _canonical_hash([record["row_identity"] for record in records])
    config_hash = _canonical_hash(asdict(config))
    expected_state = {"identity_hash": identity_hash, "config_hash": config_hash}
    if resume and state_path.is_file():
        state = read_structured(state_path)
        if any(state.get(key) != value for key, value in expected_state.items()):
            raise RuntimeError("refusing resume: cache identity/configuration differs")
        if state.get("status") == "completed" and manifest_path.is_file():
            completed = read_structured(manifest_path)
            for metadata in completed.get("shards", []):
                path = output_dir / str(metadata.get("path", ""))
                if not path.is_file() or metadata.get("sha256") != _file_sha256(path):
                    raise RuntimeError(f"resume integrity mismatch for completed shard {path}")
                if len(_load_records(path)) != int(metadata.get("count", -1)):
                    raise RuntimeError(
                        f"resume load verification failed for completed shard {path}"
                    )
            return completed
    elif not resume and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty V8 cache directory: {output_dir}")
    _atomic_json(
        state_path, {"artifact_type": V8_CACHE_SCHEMA, "status": "running", **expected_state}
    )
    shards: list[dict[str, Any]] = []
    try:
        for shard_index, start in enumerate(range(0, len(records), config.shard_size)):
            chunk = list(records[start : start + config.shard_size])
            path = output_dir / "train" / f"shard-{shard_index:05d}.pt"
            sidecar = output_dir / "train" / f"shard-{shard_index:05d}.meta.json"
            chunk_hash = _canonical_hash([record["row_identity"] for record in chunk])
            if resume and path.is_file() and sidecar.is_file():
                metadata = read_structured(sidecar)
                if metadata.get("row_identity_sha256") != chunk_hash or metadata.get(
                    "sha256"
                ) != _file_sha256(path):
                    raise RuntimeError(f"resume integrity mismatch for {path}")
                if len(_load_records(path)) != len(chunk):
                    raise RuntimeError(f"resume load verification failed for {path}")
                shards.append(metadata)
                continue
            if path.exists() or sidecar.exists():
                raise RuntimeError(
                    f"incomplete existing shard pair at {path}; use --resume only after repair"
                )
            _atomic_torch_save(chunk, path)
            roundtrip = _load_records(path)
            if len(roundtrip) != len(chunk):
                raise RuntimeError(f"roundtrip count mismatch for {path}")
            metadata = {
                "split": "train",
                "path": path.relative_to(output_dir).as_posix(),
                "count": len(chunk),
                "sha256": _file_sha256(path),
                "row_identity_sha256": chunk_hash,
                "torch_load_verified": True,
                "storage_dtype": config.storage_dtype,
            }
            _atomic_json(sidecar, metadata)
            shards.append(metadata)
    except BaseException as exc:
        _atomic_json(
            state_path,
            {
                "artifact_type": V8_CACHE_SCHEMA,
                "status": "failed",
                **expected_state,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    manifest: dict[str, Any] = {
        "artifact_type": V8_CACHE_SCHEMA,
        "format": V8_CACHE_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        **_repository_provenance(),
        "config": asdict(config),
        "config_sha256": config_hash,
        "model_input_fields": ["representation", "endpoint_us"],
        "supervision_only_fields": ["target_ttc", "sample_weight", "outer_fold"],
        "forbidden_model_input_fields": [
            "target_ttc",
            "sample_weight",
            "outer_fold",
            "row_identity",
            "common_roi_xyxy",
        ],
        "shape": [config.steps, config.channels, config.roi_size, config.roi_size],
        "split_counts": {"train": len(records)},
        "row_identity_sha256": identity_hash,
        "shards": shards,
        "train_only": True,
        "sealed_splits_opened": False,
        **dict(provenance),
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        state_path,
        {
            "artifact_type": V8_CACHE_SCHEMA,
            "status": "completed",
            **expected_state,
            "manifest_sha256": _file_sha256(manifest_path),
        },
    )
    return manifest


def materialize_scientific_recovery_v8_cache(
    *,
    eap_root: str | Path,
    garlttc_root: str | Path,
    selection_metadata_path: str | Path | None,
    output_dir: str | Path,
    config: ScientificRecoveryV8CacheConfig,
    protocol_path: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Build one V8 temporal cache exclusively from raw public-train events."""

    eap_root_path = Path(eap_root).resolve()
    garl_root_path = Path(garlttc_root).resolve()
    selection_path = (
        Path(selection_metadata_path).resolve() if selection_metadata_path is not None else None
    )
    protocol = Path(protocol_path).resolve() if protocol_path is not None else None
    plan, provenance = _planned_rows(
        eap_root=eap_root_path,
        garlttc_root=garl_root_path,
        selection_metadata_path=selection_path,
        protocol_path=protocol,
        config=config,
    )
    by_sequence: dict[tuple[str, Path], list[_PlannedRow]] = defaultdict(list)
    for row in plan:
        _assert_train_only_path(row.events_path, label="raw event HDF5")
        by_sequence[(row.sequence_id, row.events_path)].append(row)
    records: list[dict[str, Any]] = []
    for (_, event_path), sequence_rows in sorted(by_sequence.items(), key=lambda item: item[0]):
        with EAPEventReader(event_path) as reader:
            if config.representation == "timevol20":
                records.extend(_timevol_record(row, reader, config) for row in sequence_rows)
            else:
                records.extend(_exp6_records_for_sequence(sequence_rows, reader, config))
    records.sort(key=lambda record: tuple(record["row_identity"]))
    return _write_records(
        records=records,
        output_dir=Path(output_dir).resolve(),
        config=config,
        provenance=provenance,
        resume=resume,
    )


def write_scientific_recovery_v8_cache_for_testing(
    *,
    records: Sequence[dict[str, Any]],
    output_dir: str | Path,
    config: ScientificRecoveryV8CacheConfig,
    resume: bool = False,
) -> dict[str, Any]:
    """Persist already-built records for deterministic unit/integration fixtures.

    This helper is not a raw-data shortcut for experiments: its manifest marks
    ``raw_materialization=False`` and training runners must reject that flag.
    """

    _validate_records(records, config)
    storage = np.float16 if config.storage_dtype == "float16" else np.float32
    stored_records = [
        {
            **record,
            "representation": np.asarray(record["representation"], dtype=storage),
        }
        for record in records
    ]
    return _write_records(
        records=stored_records,
        output_dir=Path(output_dir).resolve(),
        config=config,
        provenance={"raw_materialization": False, "fixture_only": True},
        resume=resume,
    )


def _validate_records(
    records: Sequence[dict[str, Any]], config: ScientificRecoveryV8CacheConfig
) -> None:
    if not records:
        raise ValueError("V8 cache needs at least one record")
    expected = (config.steps, config.channels, config.roi_size, config.roi_size)
    identities: set[tuple[str, ...]] = set()
    for record in records:
        value = torch.as_tensor(record.get("representation"), dtype=torch.float32)
        if tuple(value.shape) != expected or not torch.isfinite(value).all():
            raise ValueError(f"record representation must have finite shape {expected}")
        endpoint = np.asarray(record.get("endpoint_us"), dtype=np.int64)
        if endpoint.shape != (config.steps,) or np.any(endpoint[1:] <= endpoint[:-1]):
            raise ValueError("record endpoint_us must be strictly increasing [steps]")
        identity = tuple(str(value) for value in record.get("row_identity", []))
        if len(identity) != len(GARLTTC_JOIN_KEYS) or identity in identities:
            raise ValueError("records require unique five-key row_identity values")
        identities.add(identity)
        if not math.isfinite(float(record.get("target_ttc", float("nan")))):
            raise ValueError("record target_ttc must be finite")
        if float(record.get("sample_weight", 0.0)) <= 0.0:
            raise ValueError("record sample_weight must be positive")
        boxes = np.asarray(record.get("endpoint_boxes_xyxy"), dtype=np.float32)
        if boxes.shape != (config.steps, 4) or not np.isfinite(boxes).all():
            raise ValueError("record endpoint_boxes_xyxy must be finite [steps,4]")
        heights = np.asarray(record.get("visible_heights_px"), dtype=np.float32)
        if heights.shape != (2,) or not np.isfinite(heights).all() or np.any(heights <= 0.0):
            raise ValueError("record visible_heights_px must be positive [2]")


class ScientificRecoveryV8CacheDataset(Dataset[dict[str, Any]]):
    """Lazy V8 train-cache reader with a one-shard process-local cache."""

    def __init__(self, manifest_path: str | Path, *, split: str = "train") -> None:
        if split != "train":
            raise ValueError("Scientific Recovery V8 cache exposes public train only")
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = read_structured(self.manifest_path)
        if not isinstance(self.manifest, Mapping) or not verify_artifact_hash(self.manifest):
            raise ValueError("V8 cache manifest signature is invalid")
        if (
            self.manifest.get("artifact_type") != V8_CACHE_SCHEMA
            or self.manifest.get("train_only") is not True
        ):
            raise ValueError("not a train-only Scientific Recovery V8 temporal cache")
        shape = self.manifest.get("shape")
        if not isinstance(shape, list) or len(shape) != 4:
            raise ValueError("V8 cache manifest has no [steps,channels,H,W] shape")
        self.shape = tuple(int(value) for value in shape)
        self.entries: list[tuple[Path, int]] = []
        for shard in self.manifest.get("shards", []):
            if shard.get("split") != "train":
                raise ValueError("V8 cache manifest contains a non-train shard")
            path = self.root / str(shard["path"])
            sidecar = path.with_suffix(".meta.json")
            metadata = read_structured(sidecar)
            if (
                not isinstance(metadata, Mapping)
                or not verify_artifact_hash(metadata)
                or metadata.get("sha256") != _file_sha256(path)
                or metadata.get("count") != shard.get("count")
            ):
                raise ValueError(f"V8 cache shard signature/hash is invalid: {path}")
            for index in range(int(shard["count"])):
                self.entries.append((path, index))
        if not self.entries:
            raise ValueError("V8 cache has no train records")
        self._cached_path: Path | None = None
        self._cached_records: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        groups: dict[Path, list[int]] = defaultdict(list)
        for index, (path, _) in enumerate(self.entries):
            groups[path].append(index)
        return tuple(tuple(indices) for indices in groups.values())

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, local = self.entries[index]
        if path != self._cached_path:
            self._cached_records = _load_records(path)
            self._cached_path = path
        assert self._cached_records is not None
        record = self._cached_records[local]
        return {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in record.items()
        }


def collate_scientific_recovery_v8(records: list[dict[str, Any]]) -> ScientificRecoveryV8Batch:
    """Collate generic 2/3-step V8 cache rows without exposing labels as inputs."""

    if not records:
        raise ValueError("V8 temporal collate received an empty batch")
    representations = torch.stack(
        [torch.as_tensor(row["representation"], dtype=torch.float32) for row in records]
    )
    endpoints = torch.stack(
        [torch.as_tensor(row["endpoint_us"], dtype=torch.int64) for row in records]
    )
    if representations.ndim != 5 or representations.shape[1] not in {2, 3}:
        raise ValueError("V8 representations must collate to [B,2|3,C,H,W]")
    return ScientificRecoveryV8Batch(
        representations=representations,
        endpoint_us=endpoints,
        token_id=[str(row["sample_token"]) for row in records],
        sequence_id=[str(row["sequence_id"]) for row in records],
        track_id=[str(row["track_id"]) for row in records],
        target_ttc=torch.tensor([float(row["target_ttc"]) for row in records], dtype=torch.float32),
        sample_weight=torch.tensor(
            [float(row["sample_weight"]) for row in records], dtype=torch.float32
        ),
        metadata={
            "outer_fold": torch.tensor(
                [int(row["outer_fold"]) for row in records], dtype=torch.int64
            ),
            "row_identity": [tuple(str(value) for value in row["row_identity"]) for row in records],
            "common_roi_xyxy": torch.stack(
                [torch.as_tensor(row["common_roi_xyxy"], dtype=torch.float32) for row in records]
            ),
        },
    )


def scientific_recovery_v8_model_inputs(
    batch: ScientificRecoveryV8Batch,
) -> dict[str, torch.Tensor]:
    """Return the only tensors permitted to cross the V8 model boundary."""

    return {"representations": batch.representations, "endpoint_us": batch.endpoint_us}


__all__ = [
    "ScientificRecoveryV8CacheConfig",
    "ScientificRecoveryV8CacheDataset",
    "V8_CACHE_SCHEMA",
    "collate_scientific_recovery_v8",
    "materialize_scientific_recovery_v8_cache",
    "scientific_recovery_v8_model_inputs",
    "write_scientific_recovery_v8_cache_for_testing",
]
