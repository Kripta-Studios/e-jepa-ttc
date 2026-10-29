"""Causal, label-free raw-history metadata for the independent TRAIN24 fit."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pandas.testing import assert_frame_equal

from .lineage import _load_rotation_role_unverified, verify_independent_contract
from .population import canonical_sha256, file_sha256, identity_sha256

ENDPOINT_SPAN_US, STEP_US, PACKET_ENDPOINTS = 600_000, 20_000, 31
RAW_OBSERVATION_SPAN_US = ENDPOINT_SPAN_US + STEP_US
QUERY_COUNT, GROUP_COUNT = 53_047, 24
SCHEMA = "independent_phase_raw_history_index_v4"
EXPECTED_FOLD_SHA256 = "ef67d27078a978cdfb94f0f99f660a44d64079fe5e9e2c1887841e751b6d81a4"
ROLE_KEYS = tuple(f"rotation_{i}_{r}" for i in range(6) for r in ("producer", "refiner"))
INDEX_COLUMNS = [
    "fit_ordinal",
    "sample_token",
    "group_id",
    "query_label_clock_us",
    "event_history_anchor_us",
    "sample_available_us",
    "endpoint_history_start_us",
    "observation_source_start_us",
    "packet_endpoints_us",
    "endpoint_transition_windows_us",
    "observation_windows_us",
    "events_path",
    "rgb_member_paths",
    "rgb_exposure_start_us",
    "rgb_exposure_end_us",
    "frame_timestamps_us",
    "public_source_ordinal",
    "sequence_id",
    "track_id",
    "public_track_id",
]


def _safe_path(value: object, prefix: tuple[str, ...]) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or path.parts[: len(prefix)] != prefix:
        raise ValueError(f"Unsafe or cross-sequence raw path: {text!r}")
    return text


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in (
        "packet_endpoints_us",
        "endpoint_transition_windows_us",
        "observation_windows_us",
        "rgb_member_paths",
        "rgb_exposure_start_us",
        "rgb_exposure_end_us",
        "frame_timestamps_us",
    ):
        result[col] = result[col].map(lambda x: np.asarray(x, dtype=object).tolist())
    return result.reset_index(drop=True)


def validate_history_frame(frame: pd.DataFrame) -> None:
    """Validate identity, clocks, path safety and 30 half-open intervals."""
    integers = (
        "fit_ordinal",
        "query_label_clock_us",
        "event_history_anchor_us",
        "sample_available_us",
        "endpoint_history_start_us",
        "observation_source_start_us",
        "public_source_ordinal",
    )
    if list(frame.columns) != INDEX_COLUMNS or len(frame) != QUERY_COUNT:
        raise ValueError("Raw history schema/inventory differs")
    if any(not pd.api.types.is_integer_dtype(frame[c]) for c in integers):
        raise ValueError("Raw history integer dtype differs")
    if (
        frame.fit_ordinal.tolist() != list(range(QUERY_COUNT))
        or frame.sample_token.astype(str).duplicated().any()
        or frame.public_source_ordinal.duplicated().any()
        or len(set(frame.group_id.astype(str))) != GROUP_COUNT
        or not (frame.group_id.astype(str) == frame.sequence_id.astype(str)).all()
        or frame.isna().any().any()
    ):
        raise ValueError("Raw history identity inventory differs")
    for row in frame.to_dict("records"):
        sequence = str(row["sequence_id"])
        endpoints = [int(x) for x in row["packet_endpoints_us"]]
        transitions = [[int(x) for x in pair] for pair in row["endpoint_transition_windows_us"]]
        observations = [[int(x) for x in pair] for pair in row["observation_windows_us"]]
        starts, ends = (
            [int(x) for x in row[c]] for c in ("rgb_exposure_start_us", "rgb_exposure_end_us")
        )
        members = [_safe_path(x, ("rgb",)) for x in row["rgb_member_paths"]]
        event_path = _safe_path(row["events_path"], ("data", "train", sequence))
        frames = [int(x) for x in row["frame_timestamps_us"]]
        if (
            len(endpoints) != 31
            or len(set(endpoints)) != 31
            or endpoints != sorted(endpoints)
            or len(transitions) != 30
            or len(observations) != 31
            or endpoints[0] != int(row["endpoint_history_start_us"])
            or observations[0][0] != int(row["observation_source_start_us"])
            or endpoints[-1] != int(row["event_history_anchor_us"])
            or endpoints[-1] - endpoints[0] != ENDPOINT_SPAN_US
            or transitions != [[endpoints[i], endpoints[i + 1]] for i in range(30)]
            or observations != [[endpoint - STEP_US, endpoint] for endpoint in endpoints]
            or any(b - a != STEP_US for a, b in transitions + observations)
            or len(members) != 2
            or len(set(members)) != 2
            or len(starts) != 2
            or len(ends) != 2
            or any(a < 0 or b < a for a, b in zip(starts, ends, strict=True))
            or starts[-1] != endpoints[-1]
            or int(row["sample_available_us"]) != ends[-1]
            or int(row["sample_available_us"]) < endpoints[-1]
            or int(row["observation_source_start_us"]) < 0
            or len(frames) != 2
            or frames[0] >= frames[1]
            or frames[-1] != int(row["query_label_clock_us"])
            or event_path != f"data/train/{sequence}/events.h5"
        ):
            raise ValueError("Raw history clocks/windows/exposure/path bindings differ")


def _sources(fold: Path) -> tuple[Path, Path, Path]:
    source_path = (fold.parent / "SOURCE_RECEIPT.json").resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))["sources"]
    return (
        source_path,
        Path(source["garl_data"]["path"]).resolve(strict=True),
        Path(source["eap_data"]["path"]).resolve(strict=True),
    )


def _derive(fold_contract_path: Path) -> pd.DataFrame:
    fold = fold_contract_path.resolve(strict=True)
    verified = verify_independent_contract(fold)
    if verified["fold_contract_sha256"] != EXPECTED_FOLD_SHA256:
        raise ValueError("Raw history requires pinned TRAIN24 Fold V2")
    _, garl_path, eap_path = _sources(fold)
    population = pd.read_parquet(fold.parent / "population.parquet")
    groups = sorted(population.group_id.astype(str).unique())
    garl = (
        ds.dataset(garl_path)
        .to_table(
            columns=[
                "sequence_id",
                "sample_token",
                "event_windows_us",
                "rgb_member_paths",
                "events_path",
                "frame_timestamps_us",
            ],
            filter=ds.field("sequence_id").isin(groups),
        )
        .to_pandas()
    )
    eap = (
        ds.dataset(eap_path)
        .to_table(
            columns=[
                "sequence_id",
                "rgb_member_path",
                "events_path",
                "rgb_exposure_start_timestamp_us",
                "rgb_exposure_end_timestamp_us",
            ],
            filter=ds.field("sequence_id").isin(groups),
        )
        .to_pandas()
    )
    if (
        garl.sample_token.astype(str).duplicated().any()
        or eap.duplicated(["sequence_id", "rgb_member_path"]).any()
    ):
        raise ValueError("Raw source identities are not unique")
    raw = garl.set_index("sample_token", drop=False)
    media = {
        (str(r.sequence_id), str(r.rgb_member_path)): (
            int(r.rgb_exposure_start_timestamp_us),
            int(r.rgb_exposure_end_timestamp_us),
            str(r.events_path),
        )
        for r in eap.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for query in population.to_dict("records"):
        token, sequence = str(query["sample_token"]), str(query["sequence_id"])
        if token not in raw.index:
            raise ValueError("Fold query absent from raw Garl")
        row = raw.loc[token]
        if (
            isinstance(row, pd.DataFrame)
            or str(row.sequence_id) != sequence
            or str(query["group_id"]) != sequence
        ):
            raise ValueError("Raw Garl/Fold identity differs")
        windows = [[int(x) for x in pair] for pair in row.event_windows_us]
        members = [_safe_path(x, ("rgb",)) for x in row.rgb_member_paths]
        frames = [int(x) for x in row.frame_timestamps_us]
        event_path = _safe_path(row.events_path, ("data", "train", sequence))
        if len(windows) != 2 or len(members) != 2 or len(set(members)) != 2:
            raise ValueError("Garl two-window/two-member contract differs")
        try:
            exposures = [media[(sequence, member)] for member in members]
        except KeyError as error:
            raise ValueError("Garl RGB member absent from eAP") from error
        starts, ends = [x[0] for x in exposures], [x[1] for x in exposures]
        eap_events = [_safe_path(x[2], ("data", "train", sequence)) for x in exposures]
        anchor = windows[-1][1]
        if (
            [w[1] for w in windows] != starts
            or windows[0][1] != windows[1][0]
            or any(a < 0 or b <= a for a, b in windows)
            or any(b < a for a, b in zip(starts, ends, strict=True))
            or event_path != f"data/train/{sequence}/events.h5"
            or eap_events != [event_path, event_path]
            or len(frames) != 2
            or frames[0] >= frames[1]
            or frames[-1] != int(query["timestamp_us"])
            or anchor < RAW_OBSERVATION_SPAN_US
        ):
            raise ValueError("Garl/eAP clock, member or event binding differs")
        endpoints = np.arange(anchor - ENDPOINT_SPAN_US, anchor + STEP_US, STEP_US, dtype=np.int64)
        records.append(
            {
                "fit_ordinal": int(query["fit_ordinal"]),
                "sample_token": token,
                "group_id": str(query["group_id"]),
                "query_label_clock_us": int(query["timestamp_us"]),
                "event_history_anchor_us": anchor,
                "sample_available_us": ends[-1],
                "endpoint_history_start_us": int(endpoints[0]),
                "observation_source_start_us": int(endpoints[0] - STEP_US),
                "packet_endpoints_us": endpoints.tolist(),
                "endpoint_transition_windows_us": np.column_stack(
                    (endpoints[:-1], endpoints[1:])
                ).tolist(),
                "observation_windows_us": np.column_stack(
                    (endpoints - STEP_US, endpoints)
                ).tolist(),
                "events_path": event_path,
                "rgb_member_paths": members,
                "rgb_exposure_start_us": starts,
                "rgb_exposure_end_us": ends,
                "frame_timestamps_us": frames,
                "public_source_ordinal": int(query["public_source_ordinal"]),
                "sequence_id": sequence,
                "track_id": str(query["track_id"]),
                "public_track_id": str(query["public_track_id"]),
            }
        )
    frame = (
        pd.DataFrame(records, columns=INDEX_COLUMNS)
        .sort_values("fit_ordinal", kind="stable")
        .reset_index(drop=True)
    )
    validate_history_frame(frame)
    return frame


def _inventory() -> set[str]:
    return {
        "RECEIPT.json",
        "raw_history_index.parquet",
        *(f"roles/{key}.parquet" for key in ROLE_KEYS),
    }


def build_raw_history_index(fold_contract_path: Path, output_dir: Path) -> dict[str, Any]:
    """Write once and self-verify against physical raw sources before returning."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    fold = fold_contract_path.resolve(strict=True)
    frame, source_path = _derive(fold), _sources(fold)[0]
    output_dir.mkdir(parents=True)
    index = output_dir / "raw_history_index.parquet"
    frame.to_parquet(index, index=False)
    roles: dict[str, Any] = {}
    for rotation in range(6):
        for role in ("producer", "refiner"):
            key = f"rotation_{rotation}_{role}"
            view = _load_rotation_role_unverified(fold, rotation=rotation, role=role)
            target = output_dir / "roles" / f"{key}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            view.to_parquet(target, index=False)
            roles[key] = {
                "path": target.relative_to(output_dir).as_posix(),
                "sha256": file_sha256(target),
                "rows": len(view),
                "ordered_sample_token_sha256": identity_sha256(
                    view.sample_token.astype(str).tolist()
                ),
            }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RAW_METADATA_ONLY_NO_EVENT_READS",
        "fold_contract_path": str(fold),
        "fold_contract_sha256": file_sha256(fold),
        "source_receipt_path": str(source_path),
        "source_receipt_sha256": file_sha256(source_path),
        "index_path": index.name,
        "index_sha256": file_sha256(index),
        "queries": QUERY_COUNT,
        "groups": GROUP_COUNT,
        "endpoint_span_us": ENDPOINT_SPAN_US,
        "raw_observation_span_us": RAW_OBSERVATION_SPAN_US,
        "step_us": STEP_US,
        "packet_endpoints": 31,
        "packet_intervals": 30,
        "observation_packets": 31,
        "interval_semantics": "HALF_OPEN_[START,END)",
        "runtime_materialization": "INITIALIZE_600MS_PLUS_FINAL_20MS",
        "anchor_semantics": "LAST_GARL_EVENT_WINDOW_END_EQUALS_CURRENT_EAP_EXPOSURE_START",
        "availability_semantics": "CURRENT_EAP_EXPOSURE_END",
        "label_clock_semantics": "GARL_FRAME_TIMESTAMP_DISTINCT_FROM_SENSOR_CLOCK",
        "sample_token_set_sha256": identity_sha256(
            frame.sample_token.astype(str).tolist(), ordered=False
        ),
        "roles": roles,
        "inventory": sorted(_inventory()),
        "ttc_values_read": False,
        "event_h5_opened": False,
        "stream_bounds_checked": False,
        "stream_h5_bounds_receipt": None,
        "stream_h5_hash_receipt": None,
        "training_admitted": False,
        "training_rejection": "MISSING_STREAM_H5_BOUNDS_AND_HASH_RECEIPTS",
        "learned_parents": [],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (output_dir / "RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_raw_history_index(output_dir / "RECEIPT.json")
    return receipt


def verify_raw_history_index(receipt_path: Path) -> dict[str, Any]:
    """Rederive all rows and compare all 12 producer/refiner views exactly."""
    path = receipt_path.resolve(strict=True)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    payload, claimed = dict(receipt), receipt.get("receipt_sha256")
    payload.pop("receipt_sha256", None)
    fold = Path(str(receipt.get("fold_contract_path", ""))).resolve(strict=True)
    source = Path(str(receipt.get("source_receipt_path", ""))).resolve(strict=True)
    if (
        claimed != canonical_sha256(payload)
        or receipt.get("schema") != SCHEMA
        or receipt.get("fold_contract_sha256") != EXPECTED_FOLD_SHA256
        or file_sha256(fold) != EXPECTED_FOLD_SHA256
        or source != (fold.parent / "SOURCE_RECEIPT.json").resolve(strict=True)
        or file_sha256(source) != receipt.get("source_receipt_sha256")
        or receipt.get("queries") != QUERY_COUNT
        or receipt.get("groups") != GROUP_COUNT
        or receipt.get("status") != "RAW_METADATA_ONLY_NO_EVENT_READS"
        or receipt.get("endpoint_span_us") != ENDPOINT_SPAN_US
        or receipt.get("raw_observation_span_us") != RAW_OBSERVATION_SPAN_US
        or receipt.get("step_us") != STEP_US
        or receipt.get("packet_endpoints") != PACKET_ENDPOINTS
        or receipt.get("packet_intervals") != PACKET_ENDPOINTS - 1
        or receipt.get("observation_packets") != PACKET_ENDPOINTS
        or receipt.get("interval_semantics") != "HALF_OPEN_[START,END)"
        or receipt.get("runtime_materialization") != "INITIALIZE_600MS_PLUS_FINAL_20MS"
        or receipt.get("anchor_semantics")
        != "LAST_GARL_EVENT_WINDOW_END_EQUALS_CURRENT_EAP_EXPOSURE_START"
        or receipt.get("availability_semantics") != "CURRENT_EAP_EXPOSURE_END"
        or receipt.get("label_clock_semantics") != "GARL_FRAME_TIMESTAMP_DISTINCT_FROM_SENSOR_CLOCK"
        or receipt.get("training_admitted") is not False
        or receipt.get("stream_bounds_checked") is not False
        or receipt.get("stream_h5_bounds_receipt") is not None
        or receipt.get("stream_h5_hash_receipt") is not None
        or receipt.get("event_h5_opened") is not False
        or receipt.get("ttc_values_read") is not False
        or receipt.get("learned_parents") != []
        or receipt.get("training_rejection") != "MISSING_STREAM_H5_BOUNDS_AND_HASH_RECEIPTS"
        or set(receipt.get("roles", {})) != set(ROLE_KEYS)
        or receipt.get("inventory") != sorted(_inventory())
    ):
        raise ValueError("Raw history receipt/pins/admission state differs")
    physical = {
        p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*") if p.is_file()
    }
    if physical != _inventory():
        raise ValueError("Physical raw history inventory differs")
    index = (path.parent / str(receipt["index_path"])).resolve(strict=True)
    if index.parent != path.parent or file_sha256(index) != receipt.get("index_sha256"):
        raise ValueError("Raw history index path/bytes changed")
    observed, expected = pd.read_parquet(index), _derive(fold)
    validate_history_frame(observed)
    try:
        assert_frame_equal(_normalise(observed), _normalise(expected), check_dtype=True)
    except AssertionError as error:
        raise ValueError("Raw history does not exactly rederive from raw metadata") from error
    if receipt.get("sample_token_set_sha256") != identity_sha256(
        observed.sample_token.astype(str).tolist(), ordered=False
    ):
        raise ValueError("Raw history token set differs")
    for rotation in range(6):
        for role in ("producer", "refiner"):
            key, ref = (
                f"rotation_{rotation}_{role}",
                receipt["roles"][f"rotation_{rotation}_{role}"],
            )
            relative = PurePosixPath(str(ref.get("path", "")))
            if (
                relative != PurePosixPath("roles") / f"{key}.parquet"
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise ValueError("Unsafe role path")
            role_path = (path.parent / Path(*relative.parts)).resolve(strict=True)
            actual = pd.read_parquet(role_path).reset_index(drop=True)
            wanted = _load_rotation_role_unverified(fold, rotation=rotation, role=role).reset_index(
                drop=True
            )
            if (
                file_sha256(role_path) != ref.get("sha256")
                or len(actual) != int(ref.get("rows", -1))
                or ref.get("ordered_sample_token_sha256")
                != identity_sha256(wanted.sample_token.astype(str).tolist())
            ):
                raise ValueError("Role inventory/reference differs")
            try:
                assert_frame_equal(actual, wanted, check_dtype=True)
            except AssertionError as error:
                raise ValueError("Role identities differ from Fold V2") from error
    return {"status": "PASS", "queries": QUERY_COUNT, "groups": GROUP_COUNT, "roles": 12}


def load_training_history(
    receipt_path: Path, h5_admission_path: Path | None = None
) -> pd.DataFrame:
    """Fail closed until a separate physical H5 bounds/hash admission exists."""
    receipt = json.loads(receipt_path.resolve(strict=True).read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("training_admitted") is not False
        or receipt.get("stream_bounds_checked") is not False
        or receipt.get("stream_h5_bounds_receipt") is not None
        or receipt.get("stream_h5_hash_receipt") is not None
    ):
        raise RuntimeError("Raw history admission declaration is invalid")
    if h5_admission_path is None:
        raise RuntimeError(
            "Raw history V4 is metadata-only; no verified stream-H5 bounds/hash admission exists"
        )
    from .h5_admission import verify_h5_admission

    admitted = verify_h5_admission(h5_admission_path)
    admission = json.loads(h5_admission_path.read_text(encoding="utf-8"))
    if (
        admitted.get("status") != "PASS"
        or Path(admission["raw_history_receipt_path"]).resolve(strict=True)
        != receipt_path.resolve(strict=True)
        or admission["raw_history_receipt_sha256"] != file_sha256(receipt_path)
    ):
        raise RuntimeError("H5 admission is not bound to this exact raw-history V4 receipt")
    return pd.read_parquet(receipt_path.parent / str(receipt["index_path"]))


__all__ = [
    "build_raw_history_index",
    "load_training_history",
    "validate_history_frame",
    "verify_raw_history_index",
]
