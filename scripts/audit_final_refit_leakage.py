"""Regenerate the lightweight TRAIN40/Test12 causality and split audit.

This audit reads only JSON and selected Parquet metadata columns.  It never
imports Torch, decodes media, opens event HDF5 files, or runs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTOR = REPOSITORY / "artifacts/final_refit_20261017/RGB_SOURCE_DESCRIPTOR.json"
DEFAULT_TEST_INPUTS = (
    REPOSITORY.parent
    / "e-jepa-ttc-v16-multimodal-validation"
    / "docs/multimodal_results_20261012/artifacts/multimodal_research_20261012"
    / "test12_phase_only/INPUTS.json"
)
DEFAULT_OUTPUT = REPOSITORY / "artifacts/final_refit_20261017/audit/CAUSALITY_AND_SPLITS.json"
ROLES = ("P", "H", "V")
MAX_HISTORY_US = 650_000
RGB_INTERVAL_TOLERANCE_US = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_from(owner: Path, raw: str) -> Path:
    candidate = Path(raw)
    return (candidate if candidate.is_absolute() else owner.parent / candidate).resolve(strict=True)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:], strict=False))


def nested_int_lists(value: Any) -> list[list[int]]:
    outer = as_list(value)
    if not outer:
        return []
    if not isinstance(outer[0], (list, tuple)) and not hasattr(outer[0], "tolist"):
        return [[int(item) for item in outer]]
    return [[int(item) for item in as_list(group)] for group in outer]


def duplicate_count(values: Iterable[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def recursive_forbidden_keys(value: object) -> list[str]:
    """Find likely supervision fields; paths and metadata names are excluded."""
    found: list[str] = []
    exact = {
        "target",
        "target_phase",
        "target_ttc",
        "target_ttc_s",
        "ttc",
        "gt_ttc",
        "ground_truth",
        "truth_ttc_seconds",
        "label",
        "labels",
        "depth",
        "distance",
        "relative_velocity",
    }

    def visit(item: object, prefix: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                name = str(key).lower()
                path = f"{prefix}.{key}" if prefix else str(key)
                if name in exact or name.startswith(("gt_", "ground_truth_")):
                    found.append(path)
                visit(child, path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{prefix}[{index}]")

    visit(value, "")
    return found


def audit_train(descriptor_path: Path) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("schema") != "final_refit_rgb_source_descriptor_v1":
        raise ValueError("Unexpected RGB source descriptor schema")

    columns = [
        "source_ordinal",
        "sample_token",
        "sequence_id",
        "group_id",
        "track_id",
        "query_time_us",
        "cutoff_us",
        "producer_times_us",
        "producer_available_us",
        "producer_sensor_times_us",
        "observation_times_us",
        "observation_available_us",
        "observation_sensor_times_us",
        "observation_anchor_us",
        "actual_span_us",
    ]
    all_tokens: list[str] = []
    all_sequences: list[str] = []
    all_groups: list[str] = []
    all_ordinals: list[int] = []
    track_sequences: dict[str, set[str]] = defaultdict(set)
    role_groups: dict[str, set[str]] = {}
    role_tokens: dict[str, set[str]] = {}
    sources: dict[str, Any] = {}
    violations: Counter[str] = Counter()
    interval_differences: list[int] = []
    rows_total = 0

    for role in ROLES:
        manifest_path = resolve_from(descriptor_path, str(descriptor["role_manifests"][role]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows_path = resolve_from(manifest_path, str(manifest["rows_path"]))
        rows_file_sha256 = sha256(rows_path)
        if rows_file_sha256 != str(manifest["rows_sha256"]):
            violations["declared_rows_sha256_mismatch"] += 1
        table = pq.read_table(rows_path, columns=columns)
        values = table.to_pydict()
        row_count = table.num_rows
        rows_total += row_count
        if row_count != int(manifest["population_size"]):
            violations["manifest_population_mismatch"] += 1

        tokens = [str(item) for item in values["sample_token"]]
        sequences = [str(item) for item in values["sequence_id"]]
        groups = [str(item) for item in values["group_id"]]
        ordinals = [int(item) for item in values["source_ordinal"]]
        role_tokens[role] = set(tokens)
        role_groups[role] = set(groups)
        all_tokens.extend(tokens)
        all_sequences.extend(sequences)
        all_groups.extend(groups)
        all_ordinals.extend(ordinals)

        for index in range(row_count):
            query = int(values["query_time_us"][index])
            cutoff = int(values["cutoff_us"][index])
            sequence = sequences[index]
            if groups[index] != sequence:
                violations["group_sequence_mismatch"] += 1
            track = str(values["track_id"][index])
            track_sequences[track].add(sequence)
            if cutoff > query:
                violations["cutoff_after_query"] += 1

            producer_times = [int(x) for x in as_list(values["producer_times_us"][index])]
            producer_available = [int(x) for x in as_list(values["producer_available_us"][index])]
            producer_sensor = [int(x) for x in as_list(values["producer_sensor_times_us"][index])]
            if not (
                len(producer_times) == len(producer_available) == len(producer_sensor)
                and len(producer_times) in {2, 3}
            ):
                violations["producer_clock_length"] += 1
            if not strictly_increasing(producer_times):
                violations["producer_annotation_not_strict"] += 1
            if not strictly_increasing(producer_sensor):
                violations["producer_sensor_not_strict"] += 1
            if producer_available and max(producer_available) > query:
                violations["producer_available_after_query"] += 1
            if producer_times and max(producer_times) > query:
                violations["producer_annotation_after_query"] += 1
            if any(
                arrival < stamp
                for stamp, arrival in zip(producer_times, producer_available, strict=True)
            ):
                violations["producer_available_before_annotation"] += 1
            for left, right, s_left, s_right in zip(
                producer_times,
                producer_times[1:],
                producer_sensor,
                producer_sensor[1:],
                strict=False,
            ):
                difference = (right - left) - (s_right - s_left)
                interval_differences.append(difference)
                if abs(difference) > RGB_INTERVAL_TOLERANCE_US:
                    violations["producer_cross_clock_interval_mismatch"] += 1

            observation_time_groups = nested_int_lists(values["observation_times_us"][index])
            observation_available_groups = nested_int_lists(
                values["observation_available_us"][index]
            )
            observation_sensor_groups = nested_int_lists(
                values["observation_sensor_times_us"][index]
            )
            observation_times = [item for group in observation_time_groups for item in group]
            observation_available = [
                item for group in observation_available_groups for item in group
            ]
            observation_anchors = [int(x) for x in as_list(values["observation_anchor_us"][index])]
            if any(value > query for value in observation_times):
                violations["observation_annotation_after_query"] += 1
            if any(value > query for value in observation_available):
                violations["observation_available_after_query"] += 1
            if observation_anchors and not strictly_increasing(observation_anchors):
                violations["observation_anchors_not_strict"] += 1
            if observation_anchors and observation_anchors[-1] > query:
                violations["observation_anchor_after_query"] += 1
            if any(not strictly_increasing(group) for group in observation_sensor_groups):
                violations["observation_sensor_not_strict"] += 1
            span = int(values["actual_span_us"][index])
            if span <= 0 or span > MAX_HISTORY_US:
                violations["actual_span_out_of_range"] += 1

        sources[role] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "rows_path": str(rows_path),
            "rows_file_sha256": rows_file_sha256,
            "declared_rows_sha256": str(manifest["rows_sha256"]),
            "row_count": row_count,
            "sequence_count": len(set(sequences)),
            "group_count": len(set(groups)),
            "track_count": len(set(map(str, values["track_id"]))),
            "sample_token_duplicates": duplicate_count(tokens),
        }

    overlap_by_role = {
        f"{left}_{right}": {
            "sample_tokens": len(role_tokens[left] & role_tokens[right]),
            "groups": len(role_groups[left] & role_groups[right]),
        }
        for index, left in enumerate(ROLES)
        for right in ROLES[index + 1 :]
    }
    reused_tracks = {
        track: sorted(sequences)
        for track, sequences in track_sequences.items()
        if len(sequences) > 1
    }
    ordinal_unique = len(set(all_ordinals))
    ordinal_complete = sorted(all_ordinals) == list(range(rows_total))
    if rows_total != 88_744:
        violations["train40_row_count"] += abs(rows_total - 88_744) or 1
    train_duplicates = duplicate_count(all_tokens)
    if train_duplicates:
        violations["sample_token_duplicates"] += train_duplicates
    if reused_tracks:
        violations["tracks_reused_across_sequences"] += len(reused_tracks)
    if not ordinal_complete:
        violations["source_ordinals_not_exact_zero_based_union"] += 1
    for overlap in overlap_by_role.values():
        if overlap["sample_tokens"]:
            violations["role_sample_token_overlap"] += int(overlap["sample_tokens"])
        if overlap["groups"]:
            violations["role_group_overlap"] += int(overlap["groups"])
    teacher_manifest_path = resolve_from(descriptor_path, str(descriptor["teacher_manifest"]))
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    canonical_rows_path = (teacher_manifest_path.parent / "TRAIN40_ROWS.parquet").resolve(
        strict=True
    )
    canonical = pq.read_table(
        canonical_rows_path, columns=["sample_token", "sequence_id"]
    ).to_pydict()
    canonical_pairs = set(
        zip(
            map(str, canonical["sample_token"]),
            map(str, canonical["sequence_id"]),
            strict=True,
        )
    )
    union_pairs = set(zip(all_tokens, all_sequences, strict=True))
    canonical_rows_file_sha256 = sha256(canonical_rows_path)
    canonical_rows_match = canonical_pairs == union_pairs and len(canonical_pairs) == rows_total
    if not canonical_rows_match:
        violations["canonical_train40_token_sequence_mismatch"] += 1
    if canonical_rows_file_sha256 != str(teacher_manifest["rows_sha256"]):
        violations["teacher_declared_rows_sha256_mismatch"] += 1
    if any(
        str(
            json.loads(
                resolve_from(descriptor_path, str(descriptor["role_manifests"][role])).read_text(
                    encoding="utf-8"
                )
            )["source_sha256"]
        )
        != canonical_rows_file_sha256
        for role in ROLES
    ):
        violations["role_source_sha256_not_canonical_train40"] += 1
    summary = {
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": sha256(descriptor_path),
        "sources": sources,
        "row_count": rows_total,
        "sample_token_count": len(set(all_tokens)),
        "sample_token_duplicates": train_duplicates,
        "sequence_count": len(set(all_sequences)),
        "group_count": len(set(all_groups)),
        "track_count": len(track_sequences),
        "tracks_reused_across_sequences": len(reused_tracks),
        "tracks_reused_examples": dict(list(sorted(reused_tracks.items()))[:10]),
        "source_ordinal_unique_count": ordinal_unique,
        "source_ordinals_exact_zero_based_union": ordinal_complete,
        "canonical_train40": {
            "teacher_manifest_path": str(teacher_manifest_path),
            "teacher_manifest_sha256": sha256(teacher_manifest_path),
            "rows_path": str(canonical_rows_path),
            "rows_file_sha256": canonical_rows_file_sha256,
            "declared_rows_sha256": str(teacher_manifest["rows_sha256"]),
            "row_count": len(canonical["sample_token"]),
            "token_sequence_pairs_exactly_match_role_union": canonical_rows_match,
        },
        "rgb_cross_clock_interval_difference_us": {
            "minimum": min(interval_differences) if interval_differences else None,
            "maximum": max(interval_differences) if interval_differences else None,
            "tolerance": RGB_INTERVAL_TOLERANCE_US,
        },
        "role_overlap": overlap_by_role,
        "violations": dict(sorted(violations.items())),
    }
    return summary, set(all_tokens), set(all_sequences), set(all_groups)


def _sequence_from_event_path(raw_path: str) -> str | None:
    parts = list(PureWindowsPath(raw_path).parts)
    lowered = [part.lower() for part in parts]
    if "test" not in lowered:
        return None
    index = lowered.index("test")
    return parts[index + 1] if index + 1 < len(parts) else None


def _rgb_owner(observation: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    sequences: set[str] = set()
    tracks: set[str] = set()
    for frame_id, shard in zip(
        observation.get("frame_ids", []),
        observation.get("producer_shards", []),
        strict=False,
    ):
        prefix = str(frame_id).split("|", 1)[0]
        tracks.add(prefix)
        parts = str(shard).replace("\\", "/").split("/")
        if "test" in parts and parts.index("test") + 1 < len(parts):
            sequences.add(parts[parts.index("test") + 1])
    return sequences, tracks


def audit_test(inputs_path: Path) -> tuple[dict[str, Any], set[str], set[str]]:
    payload = json.loads(inputs_path.read_text(encoding="utf-8"))
    queries = payload["queries"]
    event_observations = payload["event_observations"]
    rgb_observations = payload["rgb_observations"]
    aliases = {str(key): str(value) for key, value in payload["physical_aliases"].items()}
    violations: Counter[str] = Counter()
    query_tokens: list[str] = []
    query_sequences: list[str] = []
    query_tracks: dict[str, set[str]] = defaultdict(set)
    event_refs: Counter[str] = Counter()
    rgb_refs: Counter[str] = Counter()
    event_history_lengths: Counter[int] = Counter()
    rgb_history_lengths: Counter[int] = Counter()
    interval_differences: list[int] = []
    max_event_support = 0
    max_rgb_support = 0
    by_token = {str(query["sample_token"]): query for query in queries}

    for query in queries:
        token = str(query["sample_token"])
        sequence = str(query["sequence_id"])
        track = str(query["track_id"])
        query_time = int(query["query_time_us"])
        query_tokens.append(token)
        query_sequences.append(sequence)
        query_tracks[track].add(sequence)
        event_history = list(map(str, query["event_history"]))
        rgb_history = list(map(str, query["rgb_history"]))
        if len(event_history) != len(set(event_history)):
            violations["event_history_duplicate_reference"] += 1
        if len(rgb_history) != len(set(rgb_history)):
            violations["rgb_history_duplicate_reference"] += 1
        canonical_token = aliases.get(token)
        if canonical_token is None:
            violations["physical_alias_missing"] += 1
        elif canonical_token not in by_token:
            violations["physical_alias_unknown_canonical"] += 1
        else:
            canonical_query = by_token[canonical_token]
            if not event_history or event_history[-1] != canonical_token:
                violations["physical_alias_not_current_event"] += 1
            for field in (
                "sequence_id",
                "track_id",
                "query_time_us",
                "event_history",
                "rgb_history",
            ):
                if query[field] != canonical_query[field]:
                    violations["physical_alias_query_identity_mismatch"] += 1
        event_history_lengths[len(event_history)] += 1
        rgb_history_lengths[len(rgb_history)] += 1
        event_anchors: list[int] = []
        rgb_anchors: list[int] = []
        event_support_start = query_time
        rgb_support_start = query_time

        for observation_id in event_history:
            event_refs[observation_id] += 1
            observation = event_observations.get(observation_id)
            if observation is None:
                violations["event_history_missing_reference"] += 1
                continue
            anchor = int(observation["anchor_us"])
            available = int(observation["available_us"])
            span = int(observation["input_span_us"])
            event_anchors.append(anchor)
            event_support_start = min(event_support_start, anchor - span)
            if str(observation["sequence_id"]) != sequence:
                violations["event_sequence_owner_mismatch"] += 1
            if str(observation["track_id"]) != track:
                violations["event_track_owner_mismatch"] += 1
            if anchor > query_time or available > query_time or available < anchor:
                violations["event_clock_not_causal"] += 1
            if span <= 0 or span > MAX_HISTORY_US:
                violations["event_input_span_out_of_range"] += 1
            windows = [[int(x) for x in pair] for pair in observation["windows_us"]]
            flat = [value for pair in windows for value in pair]
            if any(len(pair) != 2 or pair[0] >= pair[1] for pair in windows):
                violations["event_window_invalid"] += 1
            if any(left[1] > right[0] for left, right in zip(windows, windows[1:], strict=False)):
                violations["event_windows_overlap_or_reverse"] += 1
            if not windows or windows[-1][1] != int(observation["sensor_anchor_us"]):
                violations["event_sensor_anchor_mismatch"] += 1
            if flat and max(flat) - min(flat) != span:
                violations["event_window_span_mismatch"] += 1
            if _sequence_from_event_path(str(observation["raw_path"])) != sequence:
                violations["event_raw_path_sequence_mismatch"] += 1

        for observation_id in rgb_history:
            rgb_refs[observation_id] += 1
            observation = rgb_observations.get(observation_id)
            if observation is None:
                violations["rgb_history_missing_reference"] += 1
                continue
            anchor = int(observation["anchor_us"])
            available = int(observation["available_us"])
            rgb_anchors.append(anchor)
            annotation = [int(x) for x in observation["producer_times_us"]]
            arrivals = [int(x) for x in observation["producer_available_us"]]
            sensor = [int(x) for x in observation["producer_sensor_times_us"]]
            if (
                len(annotation) != len(arrivals)
                or len(annotation) != len(sensor)
                or len(sensor) not in {2, 3}
            ):
                violations["rgb_producer_clock_length"] += 1
            if not strictly_increasing(annotation):
                violations["rgb_annotation_not_strict"] += 1
            if not strictly_increasing(sensor):
                violations["rgb_sensor_not_strict"] += 1
            if anchor != annotation[-1] or available != max(arrivals):
                violations["rgb_aggregate_clock_mismatch"] += 1
            if anchor > query_time or available > query_time or available < anchor:
                violations["rgb_clock_not_causal"] += 1
            if any(value > query_time for value in annotation + arrivals):
                violations["rgb_component_clock_after_query"] += 1
            if any(arrival < stamp for stamp, arrival in zip(annotation, arrivals, strict=True)):
                violations["rgb_available_before_annotation"] += 1
            for left, right, s_left, s_right in zip(
                annotation,
                annotation[1:],
                sensor,
                sensor[1:],
                strict=False,
            ):
                difference = (right - left) - (s_right - s_left)
                interval_differences.append(difference)
                if abs(difference) > RGB_INTERVAL_TOLERANCE_US:
                    violations["rgb_cross_clock_interval_mismatch"] += 1
            sequences, tracks = _rgb_owner(observation)
            if sequences != {sequence}:
                violations["rgb_sequence_owner_mismatch"] += 1
            if tracks != {track}:
                violations["rgb_track_owner_mismatch"] += 1
            rgb_support_start = min(rgb_support_start, anchor - (sensor[-1] - sensor[0]))

        if not event_anchors or not strictly_increasing(event_anchors):
            violations["event_history_not_strict"] += 1
        if not rgb_anchors or not strictly_increasing(rgb_anchors):
            violations["rgb_history_not_strict"] += 1
        event_total = query_time - event_support_start
        rgb_total = query_time - rgb_support_start
        max_event_support = max(max_event_support, event_total)
        max_rgb_support = max(max_rgb_support, rgb_total)
        if event_total > MAX_HISTORY_US:
            violations["event_total_support_over_650ms"] += 1
        if rgb_total > MAX_HISTORY_US:
            violations["rgb_total_support_over_650ms"] += 1

    orphan_event = set(event_observations) - set(event_refs)
    orphan_rgb = set(rgb_observations) - set(rgb_refs)
    reused_tracks = {k: sorted(v) for k, v in query_tracks.items() if len(v) > 1}
    forbidden = recursive_forbidden_keys(payload)
    alias_key_coverage = set(aliases) == set(query_tokens)
    alias_values_known = set(aliases.values()).issubset(set(query_tokens))
    if not alias_key_coverage:
        violations["physical_alias_key_coverage"] += 1
    if not alias_values_known:
        violations["physical_alias_value_coverage"] += 1
    query_duplicates = duplicate_count(query_tokens)
    if len(queries) != 6_762:
        violations["test12_query_count"] += abs(len(queries) - 6_762) or 1
    if query_duplicates:
        violations["query_token_duplicates"] += query_duplicates
    if len(set(query_sequences)) != 12:
        violations["test12_sequence_count"] += abs(len(set(query_sequences)) - 12) or 1
    if reused_tracks:
        violations["tracks_reused_across_sequences"] += len(reused_tracks)
    if orphan_event:
        violations["event_orphan_observation"] += len(orphan_event)
    if orphan_rgb:
        violations["rgb_orphan_observation"] += len(orphan_rgb)
    if forbidden:
        violations["forbidden_supervision_key"] += len(forbidden)
    summary = {
        "inputs_path": str(inputs_path),
        "inputs_sha256": sha256(inputs_path),
        "query_count": len(queries),
        "query_token_count": len(set(query_tokens)),
        "query_token_duplicates": query_duplicates,
        "sequence_count": len(set(query_sequences)),
        "track_count": len(query_tracks),
        "tracks_reused_across_sequences": len(reused_tracks),
        "physical_aliases": {
            "key_count": len(aliases),
            "canonical_physical_query_count": len(set(aliases.values())),
            "duplicate_logical_query_count": len(aliases) - len(set(aliases.values())),
            "keys_exactly_cover_queries": alias_key_coverage,
            "values_are_known_queries": alias_values_known,
            "policy": str(payload.get("physical_identity_policy", "")),
        },
        "event_observation_count": len(event_observations),
        "rgb_observation_count": len(rgb_observations),
        "event_orphan_observations": len(orphan_event),
        "rgb_orphan_observations": len(orphan_rgb),
        "event_history_length_distribution": {
            str(key): value for key, value in sorted(event_history_lengths.items())
        },
        "rgb_history_length_distribution": {
            str(key): value for key, value in sorted(rgb_history_lengths.items())
        },
        "maximum_event_total_support_us": max_event_support,
        "maximum_rgb_total_support_us": max_rgb_support,
        "rgb_cross_clock_interval_difference_us": {
            "minimum": min(interval_differences) if interval_differences else None,
            "maximum": max(interval_differences) if interval_differences else None,
            "tolerance": RGB_INTERVAL_TOLERANCE_US,
        },
        "forbidden_supervision_key_occurrences": len(forbidden),
        "forbidden_supervision_key_examples": forbidden[:20],
        "violations": dict(sorted(violations.items())),
    }
    return summary, set(query_tokens), set(query_sequences)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--test-inputs", type=Path, default=DEFAULT_TEST_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    descriptor = args.descriptor.resolve(strict=True)
    test_inputs = args.test_inputs.resolve(strict=True)
    train, train_tokens, train_sequences, train_groups = audit_train(descriptor)
    test, test_tokens, test_sequences = audit_test(test_inputs)

    split_checks = {
        "train_test_sample_token_overlap": len(train_tokens & test_tokens),
        "train_test_sequence_overlap": len(train_sequences & test_sequences),
        "train_test_group_id_vs_sequence_overlap": len(train_groups & test_sequences),
    }
    violations_total = (
        sum(int(value) for value in train["violations"].values())
        + sum(int(value) for value in test["violations"].values())
        + sum(split_checks.values())
    )
    payload: dict[str, Any] = {
        "schema": "final_refit_causality_and_splits_audit_v1",
        "status": "PASS" if violations_total == 0 else "FAIL",
        "scope": {
            "metadata_only": True,
            "torch_imported": False,
            "gpu_used": False,
            "media_decoded": False,
            "event_hdf5_opened": False,
            "maximum_causal_history_us": MAX_HISTORY_US,
            "clock_domain_note": (
                "RGB sensor timestamps and annotation/query timestamps are different "
                "absolute clock domains. The audit compares order and interval deltas, "
                "never their absolute values."
            ),
        },
        "source_code": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "python_executable": os.path.basename(os.sys.executable),
        },
        "train40": train,
        "test12": test,
        "split_checks": split_checks,
        "violation_count": violations_total,
    }
    payload["audit_content_sha256"] = canonical_sha256(payload)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "violation_count": violations_total,
            }
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
