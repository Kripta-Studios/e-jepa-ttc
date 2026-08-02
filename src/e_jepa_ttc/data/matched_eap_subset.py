"""Build and validate the signed, label-free matched eAP subset.

The selector in this module is deliberately boring: it projects the small
allow-list from ``data/train.parquet``, groups rows by sequence/track, and
selects complete chronological blocks.  No annotation parquet is opened and
no semantic field is read.  The resulting JSON is the only selection input
accepted by the high-resolution SSL adapter.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# This tuple is part of the protocol.  Keep it literal and ordered so a spy on
# pandas/pyarrow can prove that forbidden columns were never projected.
ALLOWED_PARQUET_COLUMNS: tuple[str, ...] = (
    "sequence_id",
    "sample_token",
    "track_id",
    "public_track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "events_path",
    "event_windows_us",
)
FORBIDDEN_FIELD_FRAGMENTS: tuple[str, ...] = (
    "annotation",
    "label",
    "ttc",
    "depth",
    "category",
    "bbox",
    "box",
    "mask",
    "rgb",
    "3d",
)
LABEL_FAMILY_PROVENANCE: dict[str, bool] = {
    "uses_ttc_labels": False,
    "uses_depth_or_3d": False,
    "uses_category_labels": False,
    "uses_boxes": False,
    "uses_masks": False,
    "uses_rgb": False,
    "uses_evttc": False,
}
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def validate_code_commit(value: object) -> str:
    """Validate the exact commit identifier bound into claim artifacts."""

    commit = str(value).strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError("code_commit must be an exact 40-hex git commit identifier.")
    return commit


def canonical_json(value: object) -> str:
    """Return the protocol's deterministic JSON representation."""

    def normalize(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: object) -> str:
    """Hash a JSON-compatible value using canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_python(value: object) -> object:
    if hasattr(value, "as_py"):
        value = value.as_py()  # type: ignore[union-attr]
    if isinstance(value, np.ndarray):
        return _as_python(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_as_python(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "(")):
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return value
    return value


def _int_or_none(value: object) -> int | None:
    value = _as_python(value)
    if value is None:
        return None
    if not isinstance(value, (int, float, str, np.integer, np.floating)):
        return None
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _window_bounds(value: object) -> tuple[int, int] | None:
    """Normalize a row's event window without inspecting any other field.

    Public eAP tables encode the interval either as ``[start, end]`` or as a
    sequence of pairs.  For the latter the first pair is the complete row
    interval (the representation code further subdivides this interval).
    """

    parsed = _as_python(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
        return None
    if all(isinstance(item, (int, float, np.number)) for item in parsed[:2]):
        start, end = _int_or_none(parsed[0]), _int_or_none(parsed[1])
    else:
        first = _as_python(parsed[0])
        last = _as_python(parsed[-1])
        if (
            not isinstance(first, (list, tuple))
            or len(first) < 2
            or not isinstance(last, (list, tuple))
            or len(last) < 2
        ):
            return None
        # eAP rows commonly carry two aligned event windows.  The frozen
        # interval is their complete span, not merely the first window.
        start, end = _int_or_none(first[0]), _int_or_none(last[1])
    if start is None or end is None or end <= start:
        return None
    return start, end


def _track_value(row: Mapping[str, Any]) -> str | None:
    for key in ("track_id", "public_track_id"):
        value = row.get(key)
        if value is None:
            continue
        text = str(_as_python(value)).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return None


def _row_identity(row: Mapping[str, Any]) -> dict[str, object]:
    """Return the identity/window payload used to derive a stable row id."""

    return {
        "sequence_id": str(row.get("sequence_id", "")),
        "sample_token": str(row.get("sample_token", "")),
        "track_id": str(row.get("track_id", "")),
        "public_track_id": str(row.get("public_track_id", "")),
        "timestamp_us": int(row["timestamp_us"]),
        "events_path": str(row["events_path"]),
        "event_windows_us": _as_python(row["event_windows_us"]),
    }


@dataclass(frozen=True)
class MatchedSubsetConfig:
    """Selection and frozen preprocessing controls for the matched subset."""

    horizons_s: tuple[float, ...] = (0.1, 0.2, 0.3)
    horizon_tolerance_s: float = 0.025
    exclusion_window_s: float = 0.02
    minimum_anchors_per_block: int = 4
    max_anchors_per_block: int = 4
    minimum_negatives: int = 2
    stage_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
    seed: int = 7
    temporal_steps: int = 5
    ssl_width: int = 320
    ssl_height: int = 192
    bins: int = 5
    batch_size: int = 2
    max_workers: int = 8
    update_budget: int = 1_000
    calibration_mode: str = "focal"
    signed_ttc_convention: str = "signed_seconds_future_minus_anchor"

    def __post_init__(self) -> None:
        if not self.horizons_s or any(float(h) <= 0.0 for h in self.horizons_s):
            raise ValueError("horizons_s must contain positive values.")
        if tuple(sorted(self.horizons_s)) != tuple(self.horizons_s):
            raise ValueError("horizons_s must be sorted chronologically.")
        if self.horizon_tolerance_s < 0.0 or self.exclusion_window_s < 0.0:
            raise ValueError("Temporal tolerance/exclusion must be non-negative.")
        if self.minimum_anchors_per_block < 4 or self.max_anchors_per_block < 4:
            raise ValueError("Blocks require at least four anchors and two negatives.")
        if self.max_anchors_per_block != 4:
            raise ValueError("The frozen pilot uses exactly four anchors per chronological block.")
        if self.minimum_negatives < 2:
            raise ValueError("NCE requires at least two negatives per anchor.")
        if not self.stage_sizes or any(int(size) <= 0 for size in self.stage_sizes):
            raise ValueError("stage_sizes must be positive.")
        if tuple(sorted(set(self.stage_sizes))) != tuple(self.stage_sizes):
            raise ValueError("stage_sizes must be strictly increasing.")
        if any(int(size) % self.max_anchors_per_block for size in self.stage_sizes):
            raise ValueError("Stage sizes must be exact multiples of max_anchors_per_block.")
        if self.temporal_steps != 5 or self.ssl_width != 320 or self.ssl_height != 192:
            raise ValueError("The frozen SSL policy is 320x192 with exactly five steps.")
        if self.bins != 5 or self.batch_size != 2 or self.max_workers > 8:
            raise ValueError("The frozen resource policy is bins=5, batch_size=2, workers<=8.")
        if self.max_workers < 0 or self.update_budget <= 0:
            raise ValueError("Invalid bounded resource controls.")


@dataclass(frozen=True)
class _ParsedRow:
    row_id: str
    sequence_id: str
    track_id: str
    sample_token: str
    timestamp_us: int
    events_path: str
    event_windows_us: object
    role: str

    @property
    def window(self) -> tuple[int, int]:
        bounds = _window_bounds(self.event_windows_us)
        if bounds is None:
            raise ValueError(f"Row {self.row_id} has an invalid event_windows_us interval.")
        return bounds


def _read_projected_parquet(source: Path) -> list[dict[str, Any]]:
    """Read exactly the allow-listed columns at Parquet projection time."""

    lowered_parts = {part.lower() for part in source.parts}
    if source.name.lower() != "train.parquet" or lowered_parts & {
        "annotation",
        "annotations",
        "label",
        "labels",
    }:
        raise ValueError(f"Forbidden annotation/label source path: {source}")
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Matched subset selection requires pandas and pyarrow.") from exc
    # Passing ``columns`` is intentional; a full parquet read would violate the
    # label-free selection contract even if the resulting frame dropped fields.
    table = pd.read_parquet(source, columns=list(ALLOWED_PARQUET_COLUMNS))
    observed = {str(column) for column in table.columns}
    missing = sorted(set(ALLOWED_PARQUET_COLUMNS) - observed)
    if missing:
        raise ValueError(f"Source parquet is missing projected columns: {missing}")
    return [dict(row) for row in table.loc[:, list(ALLOWED_PARQUET_COLUMNS)].to_dict("records")]


def _load_split(split_path: Path) -> tuple[dict[str, list[str]], str]:
    try:
        value = json.loads(split_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Split is not valid JSON: {split_path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Frozen split must be a JSON object.")
    assignments = value.get("assignments", value)
    if not isinstance(assignments, Mapping):
        raise ValueError("Frozen split must contain an assignments mapping.")
    result: dict[str, list[str]] = {}
    for role in ("train", "validation"):
        values = assignments.get(role)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Frozen split role {role!r} is missing or not a list.")
        result[role] = [str(item) for item in values]
        if not result[role]:
            raise ValueError(f"Frozen split role {role!r} is empty.")
    overlap = sorted(set(result["train"]) & set(result["validation"]))
    if overlap:
        raise ValueError(f"Frozen split roles overlap: {overlap[:5]}")
    return result, sha256_file(split_path)


def _stable_track_hash(seed: int, role: str, sequence_id: str, track_id: str) -> str:
    return hashlib.sha256(f"{seed}|{role}|{sequence_id}|{track_id}".encode()).hexdigest()


def _parsed_rows(
    rows: Sequence[Mapping[str, Any]], assignments: Mapping[str, Sequence[str]]
) -> tuple[list[_ParsedRow], dict[str, int]]:
    allowed_sequences = {
        sequence: role for role, values in assignments.items() for sequence in values
    }
    parsed: list[_ParsedRow] = []
    rejected: dict[str, int] = defaultdict(int)
    for source_row in rows:
        sequence_id = str(_as_python(source_row.get("sequence_id", ""))).strip()
        role = allowed_sequences.get(sequence_id)
        if role is None:
            rejected["sequence_not_in_frozen_split"] += 1
            continue
        track_id = _track_value(source_row)
        if track_id is None:
            rejected["missing_track_id"] += 1
            continue
        timestamp_us = _int_or_none(source_row.get("timestamp_us"))
        event_path = source_row.get("events_path")
        token = source_row.get("sample_token")
        if timestamp_us is None:
            rejected["invalid_timestamp"] += 1
            continue
        if event_path is None or token is None:
            rejected["missing_identity_or_event_path"] += 1
            continue
        if _window_bounds(source_row.get("event_windows_us")) is None:
            rejected["invalid_event_window"] += 1
            continue
        identity = _row_identity(source_row)
        parsed.append(
            _ParsedRow(
                row_id=sha256_json(identity),
                sequence_id=sequence_id,
                track_id=track_id,
                sample_token=str(_as_python(token)),
                timestamp_us=timestamp_us,
                events_path=str(_as_python(event_path)),
                event_windows_us=_as_python(source_row["event_windows_us"]),
                role=role,
            )
        )
    return parsed, dict(rejected)


def _row_payload(
    row: _ParsedRow, *, endpoints: Mapping[str, _ParsedRow], candidate_ids: Sequence[str]
) -> dict[str, Any]:
    endpoint_rows = [
        {
            "horizon_s": float(horizon),
            "row_id": endpoint.row_id,
            "sample_token": endpoint.sample_token,
            "timestamp_us": endpoint.timestamp_us,
            "events_path": endpoint.events_path,
            "event_windows_us": endpoint.event_windows_us,
            "sequence_id": endpoint.sequence_id,
            "track_id": endpoint.track_id,
        }
        for horizon, endpoint in endpoints.items()
    ]
    return {
        "row_id": row.row_id,
        "sequence_id": row.sequence_id,
        "track_id": row.track_id,
        "sample_token": row.sample_token,
        "timestamp_us": row.timestamp_us,
        "events_path": row.events_path,
        "event_windows_us": row.event_windows_us,
        "role": row.role,
        "endpoint_row_ids": {str(key): value.row_id for key, value in endpoints.items()},
        "endpoint_timestamps_us": {
            str(key): value.timestamp_us for key, value in endpoints.items()
        },
        "future_endpoints": endpoint_rows,
        "candidate_row_ids": list(candidate_ids),
    }


def _full_frame_identity(events_path: str, event_windows_us: object) -> str:
    """Normalize the identity used to keep duplicate full-frame windows together."""

    return sha256_json(
        {
            "events_path": str(events_path),
            "event_windows_us": _as_python(event_windows_us),
        }
    )


def _duplicate_units(
    rows: Sequence[_ParsedRow], endpoint_by_row: Mapping[str, Mapping[str, _ParsedRow]]
) -> list[list[_ParsedRow]]:
    """Return connected duplicate groups as inseparable chronological units."""

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        # The atomic row identity is the anchor's full-frame interval.  Future
        # endpoints naturally recur across adjacent horizons; treating those
        # overlaps as one connected component would collapse every dense track
        # into an unselectable block.  Exact duplicate anchor windows remain
        # inseparable while endpoint duplication is reported separately.
        identities = {_full_frame_identity(row.events_path, row.event_windows_us)}
        for identity in identities:
            previous = seen.get(identity)
            if previous is not None:
                union(previous, index)
            else:
                seen[identity] = index
    grouped: dict[int, list[_ParsedRow]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)
    return sorted(
        [sorted(unit, key=lambda row: (row.timestamp_us, row.row_id)) for unit in grouped.values()],
        key=lambda unit: (unit[0].timestamp_us, unit[0].row_id),
    )


def _nce_batch_metrics(
    rows: Sequence[Mapping[str, Any]],
    config: MatchedSubsetConfig,
) -> dict[str, float | int | bool]:
    """Evaluate the actual B=2,H candidate universe used by the collate/core."""

    total = 0
    valid = 0
    minimum_negatives: int | None = None
    tolerance_us = int(round(config.horizon_tolerance_s * 1_000_000.0))
    exclusion_us = int(round(config.exclusion_window_s * 1_000_000.0))
    for start in range(0, len(rows), config.batch_size):
        batch = list(rows[start : start + config.batch_size])
        if len(batch) != config.batch_size:
            continue
        for anchor in batch:
            endpoints = anchor["future_endpoints"]
            for endpoint in endpoints:
                desired = int(anchor["timestamp_us"]) + int(
                    round(float(endpoint["horizon_s"]) * 1_000_000.0)
                )
                positive_count = 0
                negative_count = 0
                for candidate in batch:
                    if (
                        candidate["sequence_id"] != anchor["sequence_id"]
                        or candidate["track_id"] != anchor["track_id"]
                    ):
                        continue
                    for candidate_endpoint in candidate["future_endpoints"]:
                        candidate_timestamp = int(candidate_endpoint["timestamp_us"])
                        positive = abs(candidate_timestamp - desired) <= tolerance_us
                        valid_candidate = (
                            positive or abs(candidate_timestamp - desired) > exclusion_us
                        )
                        if positive:
                            positive_count += 1
                        elif valid_candidate:
                            negative_count += 1
                total += 1
                valid += int(positive_count > 0 and negative_count >= config.minimum_negatives)
                minimum_negatives = (
                    negative_count
                    if minimum_negatives is None
                    else min(minimum_negatives, negative_count)
                )
    fraction = valid / max(1, total)
    return {
        "total_anchors": total,
        "valid_anchors": valid,
        "valid_anchor_fraction": fraction,
        "minimum_negatives": int(minimum_negatives or 0),
        "gate_passed": bool(
            total > 0
            and fraction >= 0.8
            and int(minimum_negatives or 0) >= config.minimum_negatives
        ),
    }


def _stage_duplicate_rates(
    stages: Sequence[Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for stage in stages:
        stage_name = str(stage["stage"])
        role_identities: dict[str, list[str]] = defaultdict(list)
        selected = {str(row_id) for row_id in stage["row_ids"]}
        for row_id in stage["row_ids"]:
            row = rows[str(row_id)]
            role = str(row["role"])
            for endpoint in row["future_endpoints"]:
                role_identities[role].append(
                    _full_frame_identity(endpoint["events_path"], endpoint["event_windows_us"])
                )
        report[stage_name] = {
            role: 1.0 - len(set(identities)) / max(1, len(identities))
            for role, identities in role_identities.items()
        }
        report[stage_name]["_selected_rows"] = float(len(selected))
    return report


def _ordered_blocks(blocks: Sequence[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    by_role_sequence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        by_role_sequence[(str(block["role"]), str(block["sequence_id"]))].append(block)
    for values in by_role_sequence.values():
        values.sort(
            key=lambda block: _stable_track_hash(
                seed,
                str(block["role"]),
                str(block["sequence_id"]),
                str(block["track_id"]),
            )
        )
    groups = sorted(by_role_sequence, key=lambda pair: (pair[0], pair[1]))
    ordered: list[dict[str, Any]] = []
    cursor = 0
    while groups:
        role, sequence = groups[cursor % len(groups)]
        values = by_role_sequence[(role, sequence)]
        if values:
            ordered.append(values.pop(0))
        groups = [pair for pair in groups if by_role_sequence[pair]]
        cursor += 1
    return ordered


def _stage_prefixes(
    blocks: Sequence[dict[str, Any]],
    sizes: Sequence[int],
    *,
    anchors_per_block: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build exact whole-block prefixes and report unavailable later stages.

    A stage is emitted only when its nominal row count can be satisfied by
    complete four-anchor blocks.  The first configured stage is a required
    pilot gate and therefore raises when unavailable; larger stages are
    omitted and recorded in the compact ``unavailable`` report.
    """

    stages: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    available_blocks = len(blocks)
    for nominal in sizes:
        block_count = int(nominal) // anchors_per_block
        if available_blocks < block_count:
            item = {
                "nominal_row_count": int(nominal),
                "required_block_count": int(block_count),
                "available_block_count": int(available_blocks),
                "available_row_count": int(available_blocks * anchors_per_block),
                "reason": "insufficient_accepted_complete_blocks",
            }
            if not stages:
                raise ValueError(
                    "Required initial matched stage is unavailable: "
                    f"nominal_size={int(nominal)}, "
                    f"available_rows={available_blocks * anchors_per_block}."
                )
            unavailable.append(item)
            continue
        chosen = list(blocks[:block_count])
        row_ids = [row_id for block in chosen for row_id in block["row_ids"]]
        if len(row_ids) != int(nominal):
            raise ValueError(
                "Matched stage construction produced a non-exact row count: "
                f"nominal={int(nominal)}, actual={len(row_ids)}."
            )
        stages.append(
            {
                "stage": f"matched_{int(nominal)}",
                "nominal_row_count": int(nominal),
                "actual_row_count": int(nominal),
                "block_ids": [str(block["block_id"]) for block in chosen],
                "row_ids": row_ids,
            }
        )
    return stages, unavailable


def _build_manifest_payload(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Sequence[str]],
    split_hash: str,
    source_path: Path,
    config: MatchedSubsetConfig,
    *,
    code_commit: str,
    diagnostic: bool,
    strict_gate: bool,
) -> dict[str, Any]:
    code_commit = validate_code_commit(code_commit)
    parsed, rejected_values = _parsed_rows(rows, assignments)
    rejected: defaultdict[str, int] = defaultdict(int, rejected_values)
    grouped: dict[tuple[str, str, str], list[_ParsedRow]] = defaultdict(list)
    for row in parsed:
        grouped[(row.role, row.sequence_id, row.track_id)].append(row)
    candidate_blocks: list[dict[str, Any]] = []
    total_candidates = 0
    total_eligible = 0
    for (role, sequence_id, track_id), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: (row.timestamp_us, row.row_id))
        timestamps = [row.timestamp_us for row in ordered]
        endpoint_by_row: dict[str, dict[str, _ParsedRow]] = {}
        eligible: list[_ParsedRow] = []
        tolerance_us = int(round(config.horizon_tolerance_s * 1_000_000.0))
        for row in ordered:
            endpoints: dict[str, _ParsedRow] = {}
            for horizon in config.horizons_s:
                desired = row.timestamp_us + int(round(horizon * 1_000_000.0))
                left = bisect_left(timestamps, desired - tolerance_us)
                right = bisect_right(timestamps, desired + tolerance_us)
                matches = ordered[left:right]
                if matches:
                    endpoints[str(horizon)] = min(
                        matches,
                        key=lambda candidate: (
                            abs(candidate.timestamp_us - desired),
                            candidate.row_id,
                        ),
                    )
            total_candidates += 1
            if len(endpoints) != len(config.horizons_s):
                rejected["missing_future_endpoint"] += 1
                continue
            endpoint_by_row[row.row_id] = endpoints
            eligible.append(row)
        total_eligible += len(eligible)
        if len(ordered) < config.minimum_anchors_per_block:
            rejected["track_shorter_than_minimum_block"] += len(ordered)
            continue
        units = _duplicate_units(eligible, endpoint_by_row)
        current: list[_ParsedRow] = []
        units_for_blocks: list[list[_ParsedRow]] = []
        for unit in units:
            if len(unit) > config.max_anchors_per_block:
                rejected["duplicate_group_exceeds_block_size"] += len(unit)
                continue
            if current and len(current) + len(unit) > config.max_anchors_per_block:
                if len(current) == config.max_anchors_per_block:
                    units_for_blocks.append(current)
                else:
                    rejected["remainder_shorter_than_block"] += len(current)
                current = []
            current.extend(unit)
            if len(current) == config.max_anchors_per_block:
                units_for_blocks.append(current)
                current = []
        if current:
            rejected["remainder_shorter_than_block"] += len(current)
        for block_rows_source in units_for_blocks:
            payload_rows = [
                _row_payload(
                    row,
                    endpoints=endpoint_by_row[row.row_id],
                    candidate_ids=[candidate.row_id for candidate in block_rows_source],
                )
                for row in block_rows_source
            ]
            nce = _nce_batch_metrics(payload_rows, config)
            if not bool(nce["gate_passed"]):
                rejected["block_nce_gate_failed"] += len(payload_rows)
                continue
            block_id = sha256_json(
                {
                    "role": role,
                    "sequence_id": sequence_id,
                    "track_id": track_id,
                    "row_ids": [row["row_id"] for row in payload_rows],
                }
            )
            for row in payload_rows:
                row["block_id"] = block_id
            candidate_blocks.append(
                {
                    "block_id": block_id,
                    "role": role,
                    "sequence_id": sequence_id,
                    "track_id": track_id,
                    "anchor_count": len(payload_rows),
                    "row_ids": [row["row_id"] for row in payload_rows],
                    "rows": payload_rows,
                    "nce": nce,
                }
            )
    ordered_candidates = _ordered_blocks(candidate_blocks, seed=config.seed)
    max_nominal = max(config.stage_sizes)
    max_block_count = max_nominal // config.max_anchors_per_block
    candidate_prefix = ordered_candidates[:max_block_count]
    stages, unavailable_stages = _stage_prefixes(
        candidate_prefix,
        config.stage_sizes,
        anchors_per_block=config.max_anchors_per_block,
    )
    # Keep only the rows/blocks represented by the largest *emitted* stage.
    # This is the compact max-stage payload consumed by the adapter; rows from
    # a partial unavailable suffix are never silently retained.
    max_stage_block_ids = set(stages[-1]["block_ids"])
    ordered_blocks = [
        block for block in candidate_prefix if str(block["block_id"]) in max_stage_block_ids
    ]
    if len(ordered_blocks) * config.max_anchors_per_block != int(stages[-1]["nominal_row_count"]):
        raise ValueError("Largest emitted matched stage is not an exact block prefix.")
    all_rows = [row for block in ordered_blocks for row in block["rows"]]
    row_map = {str(row["row_id"]): row for row in all_rows}
    sampler_order = [str(row["row_id"]) for row in all_rows]
    stage_metrics: dict[str, Any] = {}
    for stage in stages:
        stage_blocks = [
            block for block in ordered_blocks if block["block_id"] in set(stage["block_ids"])
        ]
        stage_rows = [row for block in stage_blocks for row in block["rows"]]
        aggregate = _nce_batch_metrics(stage_rows, config)
        by_role: dict[str, Any] = {}
        for role in ("train", "validation"):
            role_rows = [row for row in stage_rows if row["role"] == role]
            by_role[role] = _nce_batch_metrics(role_rows, config)
        stage_metrics[str(stage["stage"])] = {"overall": aggregate, "by_role": by_role}
    report = {
        "source_row_count": len(rows),
        "parsed_row_count": len(parsed),
        "eligible_anchor_count": total_eligible,
        "candidate_anchor_count": total_candidates,
        "nce_anchor_coverage": float(
            stage_metrics[str(stages[-1]["stage"])]["overall"]["valid_anchor_fraction"]
            if stages and stages[-1]["row_ids"]
            else 0.0
        ),
        "minimum_negatives": int(
            stage_metrics[str(stages[-1]["stage"])]["overall"]["minimum_negatives"]
            if stages and stages[-1]["row_ids"]
            else 0
        ),
        "nce_by_stage": stage_metrics,
        "unavailable_stages": unavailable_stages,
        "duplicate_full_frame_endpoint_rate_by_stage_role": _stage_duplicate_rates(stages, row_map),
        "rejected_reasons": dict(sorted(rejected.items())),
        "per_sequence_counts": {
            sequence: sum(1 for row in all_rows if row["sequence_id"] == sequence)
            for sequence in sorted({str(row["sequence_id"]) for row in all_rows})
        },
        "per_track_counts": {
            f"{sequence}|{track}": sum(
                1 for row in all_rows if row["sequence_id"] == sequence and row["track_id"] == track
            )
            for sequence, track in sorted(
                {(str(row["sequence_id"]), str(row["track_id"])) for row in all_rows}
            )
        },
    }
    selected_track_counts = list(report["per_track_counts"].values())
    report["long_track_concentration"] = float(
        max(selected_track_counts, default=0) / max(1, len(all_rows))
    )
    if strict_gate and (
        float(report["nce_anchor_coverage"]) < 0.8
        or int(report["minimum_negatives"]) < config.minimum_negatives
    ):
        raise ValueError(
            "Matched subset NCE gate failed: "
            f"coverage={report['nce_anchor_coverage']:.3f} (need >=0.8), "
            f"minimum_negatives={report['minimum_negatives']} (need >= {config.minimum_negatives})."
        )
    config_payload = asdict(config)
    config_payload["horizons_s"] = list(config.horizons_s)
    config_payload["stage_sizes"] = list(config.stage_sizes)
    sampler_order_hash = sha256_json(sampler_order)
    # The signed dataset hash is over the projected, allowed rows.  Hashing the
    # raw parquet bytes here would make a mutation to an unprojected TTC/box
    # column alter an otherwise identical SSL selection.
    projected_source_hash = sha256_json(rows)
    selection_core = {
        "code_commit": code_commit,
        "rows": all_rows,
        "stages": stages,
        "config": config_payload,
        "split_hash": split_hash,
        "source_parquet_sha256": projected_source_hash,
        "sampler_order_hash": sampler_order_hash,
    }
    matched_manifest_hash = sha256_json(selection_core)
    payload: dict[str, Any] = {
        "artifact_type": "matched_eap_subset_v1",
        "schema_version": "matched_eap_subset_v1",
        "evidence_type": (
            "diagnostic_label_free_manifest_selection"
            if diagnostic
            else "label_free_manifest_selection"
        ),
        "code_commit": code_commit,
        "protocol_version": "matched_eap_subset_v1",
        "protocol_sha256": sha256_json(
            {"projected_columns": list(ALLOWED_PARQUET_COLUMNS), "selection_rule": "v2"}
        ),
        "created_at": "selection_time_not_recorded",
        "matched_manifest_hash": matched_manifest_hash,
        "source": {
            "dataset": "GarlTTC/eAP",
            "parquet": "data/train.parquet",
            "parquet_sha256": selection_core["source_parquet_sha256"],
            "projected_columns": list(ALLOWED_PARQUET_COLUMNS),
            "annotations_opened": False,
            "labels_path_opened": False,
        },
        "dataset_hashes": {
            "source_parquet_sha256": selection_core["source_parquet_sha256"],
            "config_sha256": sha256_json(config_payload),
            "split_sha256": split_hash,
            "sampler_order_sha256": sampler_order_hash,
        },
        "split_hash": split_hash,
        "split_assignments": {role: list(values) for role, values in assignments.items()},
        "selection_rule": "label_free_fixed_four_anchor_blocks_round_robin_v2",
        "sampler_order_hash": sampler_order_hash,
        "sampler_order": sampler_order,
        "config": config_payload,
        "freeze": {
            "modalities": ["events"],
            "ssl_input_policy": "full_frame_event_only_320x192_from_raw",
            "downstream_input_policy": "official_square_object_roi_128x128_post_ssl_only",
            "temporal_steps": config.temporal_steps,
            "interval_source": "per_row_event_windows_us_subdivided_exactly",
            "calibration_mode": config.calibration_mode,
            "signed_ttc_convention": config.signed_ttc_convention,
            "seeds": [7, 13, 23],
            "batch_size": config.batch_size,
            "max_workers": config.max_workers,
            "update_budget": config.update_budget,
        },
        "selection_report": report,
        "stages": stages,
        "blocks": ordered_blocks,
        "rows": all_rows,
        "label_family_provenance": dict(LABEL_FAMILY_PROVENANCE),
        "uses_dense_disk_cache": False,
    }
    # Keep the scientific artifact metadata self-contained without introducing
    # a second non-deterministic hash/signature cycle.
    payload["artifact_sha256"] = matched_manifest_hash
    payload["signature"] = sha256_json(payload)
    return payload


def build_matched_eap_subset(
    garlttc_root: str | Path,
    split_path: str | Path,
    *,
    config: MatchedSubsetConfig | Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    code_commit: str | None = None,
    diagnostic: bool = False,
    strict_gate: bool = True,
) -> dict[str, Any]:
    """Build a signed manifest from the projected, label-free train parquet."""

    if code_commit is None:
        raise ValueError("A validated 40-hex code_commit is required for a claim manifest.")
    code_commit = validate_code_commit(code_commit)
    root = Path(garlttc_root)
    source = root / "data" / "train.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"Expected label-free source parquet is missing: {source}")
    split = Path(split_path)
    if not split.is_file():
        raise FileNotFoundError(f"Frozen split is missing: {split}")
    assignments, split_hash = _load_split(split)
    if config is None:
        resolved = MatchedSubsetConfig()
    elif isinstance(config, MatchedSubsetConfig):
        resolved = config
    else:
        allowed = set(asdict(MatchedSubsetConfig()))
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"Unknown matched subset config keys: {unknown}")
        values = dict(config)
        if "horizons_s" in values:
            values["horizons_s"] = tuple(float(item) for item in values["horizons_s"])
        if "stage_sizes" in values:
            values["stage_sizes"] = tuple(int(item) for item in values["stage_sizes"])
        resolved = MatchedSubsetConfig(**values)
    rows = _read_projected_parquet(source)
    payload = _build_manifest_payload(
        rows,
        assignments,
        split_hash,
        source,
        resolved,
        code_commit=code_commit,
        diagnostic=diagnostic,
        strict_gate=strict_gate,
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    return payload


build_manifest = build_matched_eap_subset


@dataclass(frozen=True)
class MatchedEAPSubsetBuilder:
    """Small object wrapper for callers that keep selection config state."""

    config: MatchedSubsetConfig = field(default_factory=MatchedSubsetConfig)
    code_commit: str | None = None
    diagnostic: bool = False
    strict_gate: bool = True

    def build(
        self,
        garlttc_root: str | Path,
        split_path: str | Path,
        *,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        return build_matched_eap_subset(
            garlttc_root,
            split_path,
            config=self.config,
            output_path=output_path,
            code_commit=self.code_commit,
            diagnostic=self.diagnostic,
            strict_gate=self.strict_gate,
        )


def validate_matched_manifest(value: Mapping[str, Any], *, require_gate: bool = True) -> None:
    """Validate a loaded signed manifest without touching any dataset source."""

    if value.get("artifact_type") != "matched_eap_subset_v1":
        raise ValueError("Unexpected matched subset artifact_type.")
    code_commit = validate_code_commit(value.get("code_commit"))
    signature = value.get("signature")
    if not isinstance(signature, str) or signature != sha256_json(
        {k: v for k, v in value.items() if k != "signature"}
    ):
        raise ValueError("Matched subset signature mismatch.")
    projection = value.get("source", {}).get("projected_columns", [])
    if projection != list(ALLOWED_PARQUET_COLUMNS):
        raise ValueError("Matched subset projected_columns differ from the frozen allow-list.")
    provenance = value.get("label_family_provenance")
    if provenance != LABEL_FAMILY_PROVENANCE:
        raise ValueError("Matched subset label_family_provenance must explicitly be all false.")
    rows = value.get("rows")
    blocks = value.get("blocks")
    stages = value.get("stages")
    if (
        not isinstance(rows, list)
        or not isinstance(blocks, list)
        or not isinstance(stages, list)
        or not stages
    ):
        raise ValueError("Matched subset rows/blocks/stages are required.")
    row_map = {str(row.get("row_id")): row for row in rows if isinstance(row, Mapping)}
    if len(row_map) != len(rows) or any(not row_id for row_id in row_map):
        raise ValueError("Matched subset rows must have unique non-empty row IDs.")
    block_map = {
        str(block.get("block_id")): block for block in blocks if isinstance(block, Mapping)
    }
    if len(block_map) != len(blocks) or any(not block_id for block_id in block_map):
        raise ValueError("Matched subset blocks must have unique non-empty block IDs.")
    config = value.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("Matched subset config is required for structural validation.")
    anchors_per_block = int(config.get("max_anchors_per_block", 4))
    if anchors_per_block != 4:
        raise ValueError("Matched subset blocks must contain exactly four anchors.")
    split_assignments = value.get("split_assignments")
    if not isinstance(split_assignments, Mapping):
        raise ValueError("Matched subset split_assignments are required.")
    role_sequences = {
        str(sequence): str(role)
        for role, sequences in split_assignments.items()
        if isinstance(sequences, Sequence) and not isinstance(sequences, (str, bytes))
        for sequence in sequences
    }
    observed_block_rows: list[str] = []
    for block_id, block in block_map.items():
        row_ids = block.get("row_ids")
        if not isinstance(row_ids, list) or len(row_ids) != anchors_per_block:
            raise ValueError(f"Block {block_id} does not contain exactly four rows.")
        if len(set(str(row_id) for row_id in row_ids)) != len(row_ids):
            raise ValueError(f"Block {block_id} contains duplicate rows.")
        expected_role = str(block.get("role", ""))
        expected_sequence = str(block.get("sequence_id", ""))
        expected_track = str(block.get("track_id", ""))
        nce = block.get("nce")
        if not isinstance(nce, Mapping):
            raise ValueError(f"Block {block_id} lacks actual NCE metrics.")
        if not bool(nce.get("gate_passed", False)):
            raise ValueError(f"Block {block_id} failed the actual NCE gate.")
        if int(nce.get("minimum_negatives", 0)) < int(config.get("minimum_negatives", 2)):
            raise ValueError(f"Block {block_id} has too few NCE negatives.")
        for row_id in row_ids:
            row = row_map.get(str(row_id))
            if row is None:
                raise ValueError(f"Block {block_id} references an unknown row.")
            if str(row.get("block_id")) != block_id:
                raise ValueError(f"Row {row_id} has inconsistent block membership.")
            if (
                str(row.get("role")) != expected_role
                or str(row.get("sequence_id")) != expected_sequence
                or str(row.get("track_id")) != expected_track
            ):
                raise ValueError(
                    f"Block {block_id} has inconsistent role/sequence/track membership."
                )
            if role_sequences.get(str(row.get("sequence_id"))) != str(row.get("role")):
                raise ValueError(f"Row {row_id} violates the frozen role/split assignment.")
            observed_block_rows.append(str(row_id))
        if observed_block_rows[-len(row_ids) :] != [str(row_id) for row_id in row_ids]:
            raise ValueError(f"Block {block_id} row order is not preserved.")
    if len(set(observed_block_rows)) != len(observed_block_rows):
        raise ValueError("A row belongs to more than one matched block.")
    if set(observed_block_rows) != set(row_map):
        raise ValueError("Manifest rows and block membership differ.")
    previous_rows: list[str] = []
    stage_size_values = [int(config_size) for config_size in config.get("stage_sizes", [])]
    if not stage_size_values or stage_size_values != sorted(set(stage_size_values)):
        raise ValueError("Matched subset config stage_sizes must be strictly increasing.")
    emitted_nominals: list[int] = []
    for stage in stages:
        stage_name = str(stage.get("stage", ""))
        row_ids = [str(row_id) for row_id in stage.get("row_ids", [])]
        block_ids = [str(block_id) for block_id in stage.get("block_ids", [])]
        unknown_blocks = [block_id for block_id in block_ids if block_id not in block_map]
        if unknown_blocks:
            raise ValueError(f"Stage {stage_name} references unknown blocks: {unknown_blocks[:3]}")
        expected_rows = [
            str(row_id) for block_id in block_ids for row_id in block_map[block_id]["row_ids"]
        ]
        if row_ids != expected_rows:
            raise ValueError(f"Stage {stage_name} is not an exact ordered block prefix.")
        if previous_rows and row_ids[: len(previous_rows)] != previous_rows:
            raise ValueError("Matched subset stages are not nested ordered prefixes.")
        previous_rows = row_ids
        nominal = int(stage.get("nominal_row_count", 0))
        if nominal not in stage_size_values:
            raise ValueError(f"Stage {stage_name} nominal count is absent from config.")
        if emitted_nominals and nominal <= emitted_nominals[-1]:
            raise ValueError("Matched subset stages must be strictly increasing.")
        emitted_nominals.append(nominal)
        if int(stage.get("actual_row_count", -1)) != nominal or len(row_ids) != nominal:
            raise ValueError(
                f"Stage {stage_name} must contain exactly its nominal row count ({nominal})."
            )
    if emitted_nominals[0] != stage_size_values[0]:
        raise ValueError("The required initial matched stage is missing.")
    selection_report = value.get("selection_report", {})
    if not isinstance(selection_report, Mapping) or not isinstance(
        selection_report.get("unavailable_stages"), list
    ):
        raise ValueError("Matched subset unavailable_stages report is required.")
    unavailable_nominals = [
        int(item.get("nominal_row_count", 0))
        for item in selection_report["unavailable_stages"]
        if isinstance(item, Mapping)
    ]
    if len(unavailable_nominals) != len(selection_report["unavailable_stages"]):
        raise ValueError("Unavailable stage entries must be mappings.")
    if set(emitted_nominals) & set(unavailable_nominals) or set(emitted_nominals) | set(
        unavailable_nominals
    ) != set(stage_size_values):
        raise ValueError("Emitted/unavailable stages do not match configured stage sizes.")
    nce_by_stage = selection_report.get("nce_by_stage")
    if not isinstance(nce_by_stage, Mapping):
        raise ValueError("Per-stage NCE metrics are required.")
    for nominal in emitted_nominals:
        stage_name = f"matched_{nominal}"
        stage_metrics = nce_by_stage.get(stage_name)
        if not isinstance(stage_metrics, Mapping):
            raise ValueError(f"Missing per-stage NCE metrics for {stage_name}.")
        overall = stage_metrics.get("overall")
        by_role = stage_metrics.get("by_role")
        if not isinstance(overall, Mapping) or not bool(overall.get("gate_passed", False)):
            raise ValueError(f"Stage {stage_name} failed the actual NCE gate.")
        if not isinstance(by_role, Mapping):
            raise ValueError(f"Stage {stage_name} lacks per-role NCE metrics.")
        stage_rows = [row_map[row_id] for row_id in previous_rows[:nominal]]
        for role in ("train", "validation"):
            role_rows = [row for row in stage_rows if str(row.get("role")) == role]
            role_metric = by_role.get(role)
            if role_rows and (
                not isinstance(role_metric, Mapping)
                or not bool(role_metric.get("gate_passed", False))
            ):
                raise ValueError(f"Stage {stage_name} role {role} failed the NCE gate.")
    largest_nominal = emitted_nominals[-1]
    if len(row_map) != largest_nominal:
        raise ValueError(
            "Manifest rows must contain exactly the rows of the largest emitted stage."
        )
    if value.get("sampler_order") != previous_rows:
        raise ValueError("Sampler order must equal the maximum-stage ordered rows.")
    if value.get("sampler_order_hash") != sha256_json(previous_rows):
        raise ValueError("Sampler order hash mismatch.")
    source = value.get("source")
    source_hash = source.get("parquet_sha256") if isinstance(source, Mapping) else None
    expected_manifest_hash = sha256_json(
        {
            "code_commit": code_commit,
            "rows": rows,
            "stages": stages,
            "config": config,
            "split_hash": value.get("split_hash"),
            "source_parquet_sha256": source_hash,
            "sampler_order_hash": value.get("sampler_order_hash"),
        }
    )
    if value.get("matched_manifest_hash") != expected_manifest_hash:
        raise ValueError("Matched subset matched_manifest_hash does not match its signed contents.")
    if value.get("artifact_sha256") != value.get("matched_manifest_hash"):
        raise ValueError("Matched subset artifact_sha256 must bind matched_manifest_hash.")
    horizons = [float(horizon) for horizon in config.get("horizons_s", [])]
    if not horizons:
        raise ValueError("Manifest config horizons_s are required.")
    tolerance = float(config.get("horizon_tolerance_s", 0.0))
    for row_id, row in row_map.items():
        endpoint_map = row.get("endpoint_row_ids")
        timestamp_map = row.get("endpoint_timestamps_us")
        endpoints = row.get("future_endpoints")
        if (
            not isinstance(endpoint_map, Mapping)
            or not isinstance(timestamp_map, Mapping)
            or not isinstance(endpoints, list)
        ):
            raise ValueError(f"Row {row_id} has incomplete endpoint metadata.")
        if set(str(key) for key in endpoint_map) != {str(horizon) for horizon in horizons}:
            raise ValueError(f"Row {row_id} endpoint horizons differ from config.")
        endpoint_lookup = {
            str(endpoint.get("horizon_s")): endpoint
            for endpoint in endpoints
            if isinstance(endpoint, Mapping)
        }
        for horizon in horizons:
            key = str(horizon)
            endpoint = endpoint_lookup.get(key)
            if endpoint is None or str(endpoint.get("row_id")) != str(endpoint_map[key]):
                raise ValueError(f"Row {row_id} endpoint identity mismatch at horizon {horizon}.")
            desired = int(row["timestamp_us"]) + int(round(horizon * 1_000_000.0))
            if abs(int(endpoint["timestamp_us"]) - desired) > int(round(tolerance * 1_000_000.0)):
                raise ValueError(f"Row {row_id} endpoint timestamp is outside tolerance.")
            if str(endpoint.get("sequence_id")) != str(row["sequence_id"]) or str(
                endpoint.get("track_id")
            ) != str(row["track_id"]):
                raise ValueError(f"Row {row_id} endpoint crosses sequence/track boundaries.")
    report = value.get("selection_report", {})
    if require_gate and (
        float(report.get("nce_anchor_coverage", 0.0)) < 0.8
        or int(report.get("minimum_negatives", 0)) < 2
    ):
        raise ValueError("Matched subset NCE coverage/min-negative gate failed.")


def load_matched_manifest(path: str | Path, *, require_gate: bool = True) -> dict[str, Any]:
    """Load, verify and validate a signed matched-subset manifest."""

    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Matched subset manifest must be a JSON object.")
    validate_matched_manifest(value, require_gate=require_gate)
    return dict(value)


__all__ = [
    "ALLOWED_PARQUET_COLUMNS",
    "LABEL_FAMILY_PROVENANCE",
    "MatchedSubsetConfig",
    "MatchedEAPSubsetBuilder",
    "build_manifest",
    "build_matched_eap_subset",
    "canonical_json",
    "load_matched_manifest",
    "sha256_file",
    "sha256_json",
    "validate_code_commit",
    "validate_matched_manifest",
]
