"""Adversarial artifact tests for the independent, inference-free run verifier."""

from __future__ import annotations

import copy
import csv
import json
import struct
import zipfile

import pytest

from scripts.verify_evttc_competition_run import (
    COLUMNS,
    MODEL_NAMES,
    SEQUENCES,
    AuditError,
    audit_run,
    canonical_sha,
    file_sha,
    sensor_clocks,
    validate_admission,
    validate_temporal,
    validate_v2_plan,
)


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _repackage(run):
    entries = []
    with zipfile.ZipFile(run / "results.zip", "w") as archive:
        for sequence in SEQUENCES:
            relative = f"{sequence}/results.txt"
            path = run / "results" / relative
            archive.write(path, relative)
            entries.append(
                {
                    "sequence_id": sequence,
                    "path": relative,
                    "rows": len(path.read_text().splitlines()) - 1,
                    "sha256": file_sha(path),
                }
            )
    for name in ("summary.json", "manifest.json"):
        value = json.loads((run / name).read_text())
        value["results_zip_sha256"] = file_sha(run / "results.zip")
        if name == "summary.json":
            value["results_validation"] = {"valid": True, "errors": [], "files": entries}
        _json(run / name, value)


@pytest.fixture
def complete_run(tmp_path):
    run, data = tmp_path / "run", tmp_path / "data"
    run.mkdir()
    data.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    components = {}
    for name in MODEL_NAMES:
        path = candidate / name
        path.write_bytes(name.encode())
        components[name] = {"local_path": str(path), "sha256": file_sha(path)}
    _json(candidate / "PORTABLE_PATHS.json", {"components": components})
    source = tmp_path / "test_source.py"
    source.write_text("# unchanged fixture source\n")
    clocks, identities, raw_sources = {}, {}, {}
    counts = [50, 43, 30, 46, 55, 59, 42, 40, 51, 18]
    rows = []
    for sequence, count in zip(SEQUENCES, counts, strict=True):
        folder = data / sequence
        folder.mkdir()
        times = [300_000 + ordinal * 100_000 for ordinal in range(count)]
        clocks[sequence] = {"rgb_us": times, "event_start_us": 0, "event_end_us": times[-1] + 1}
        csv_rows = [
            [str(i), f"{timestamp / 1e6:.16f}", "1", "1", "4", "4"]
            for i, timestamp in enumerate(times)
        ]
        with (folder / "bbox.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows(csv_rows)
        names = ["bbox.csv", "data.hdf5"]
        if sequence.startswith("Slider"):
            names = ["bbox.csv", "raw.bag", "calibParams.mat"]
        for name in names[1:]:
            (folder / name).write_bytes(b"fixture sensor data")
        raw_sources[sequence] = {
            "files": {
                name: {"bytes": (folder / name).stat().st_size, "sha256": file_sha(folder / name)}
                for name in names
            }
        }
        identities[sequence] = [
            {
                "sequence_id": sequence,
                "row_index": i,
                "index": i,
                "timestamp_text": record[1],
                "anchor_us": times[i],
                "box_text": record[2:],
            }
            for i, record in enumerate(csv_rows)
        ]
        for ordinal, identity in enumerate(identities[sequence]):
            query = times[ordinal]
            prediction = {
                "status": "ok",
                "model_ttc": -2.5,
                "model_reason": "event_only_fallback",
                "event_history_count": 1,
                "rgb_history_count": 0,
                "model_phase_delta_seconds": 0.1,
                "producer_diagnostics": {
                    "event": {
                        "known_true": 0,
                        "known_total": 2,
                        "support_count": 6,
                        "support_min": 0.0,
                        "support_max": 0.0,
                    }
                },
            }
            observation = {
                "windows_us": [
                    [query - 300_000 + i * 100_000, query - 200_000 + i * 100_000] for i in range(3)
                ],
                "raw_counts": [1, 1, 1],
                "roi_counts": [0, 0, 0],
                "first_sensor_ts_us": query - 300_000,
                "anchor_us": query,
                "available_us": query,
                "delta_t_s": [0.1, 0.1],
                "admitted": True,
                "zero_support": True,
            }
            boxes = [
                {"index": item["index"], "anchor_us": item["anchor_us"], "raw_box": [1, 1, 4, 4]}
                for item in identities[sequence][max(0, ordinal - 1) : ordinal + 1]
            ]
            rows.append(
                {
                    "schema": "evttc_competition_row_v1",
                    "identity": identity,
                    "status": "ok",
                    "prediction_ttc_seconds": -2.5,
                    "cost_time_seconds": 0.125,
                    "preprocess_seconds": 0.025,
                    "model_wall_seconds": 0.1,
                    "reason": "event_only_fallback",
                    "prediction": prediction,
                    "runtime_internal": copy.deepcopy(prediction),
                    "input_details": {
                        "query_us": query,
                        "event_recording_start_us": 0,
                        "event_recording_end_us": times[-1] + 1,
                        "event_observations": [observation],
                        "rgb_observations": [],
                        "bbox_sources": boxes,
                        "startup_mode": "native_100ms_windows",
                    },
                }
            )
    freeze = {
        "sequences": list(SEQUENCES),
        "queries": 434,
        "run_mode": "inference",
        "targets_read": False,
        "training_updates": 0,
        "checkpoint_selection": False,
        "policy": {},
        "query_manifest": identities,
        "calibrations": {},
        "raw_sources": raw_sources,
        "executed_source_sha256": {source.name: file_sha(source)},
        "frozen_runtime": {"source_sha256": {}},
        "candidate_root_resolved": str(candidate),
        "model": {
            "portable_paths_sha256": file_sha(candidate / "PORTABLE_PATHS.json"),
            "components": {name: item["sha256"] for name, item in components.items()},
        },
    }
    for field in ("policy", "query_manifest", "calibrations", "raw_sources"):
        freeze[f"{field}_sha256"] = canonical_sha(freeze[field])
    _json(run / "RUN_FREEZE.json", freeze)
    freeze_hash = file_sha(run / "RUN_FREEZE.json")
    for row in rows:
        row["freeze_sha256"] = freeze_hash
    (run / "all_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _json(
        run / "summary.json",
        {
            "status": "COMPLETE",
            "freeze_sha256": freeze_hash,
            "expected_rows": 434,
            "ok_rows": 434,
            "recorded_rows": 434,
            "null_rows": 0,
            "zero_support_rows": 434,
            "results_zip": "results.zip",
            "producer_known_mask_diagnostics": {
                "event": {"known_true": 0, "known_total": 868, "support_count": 2604}
            },
        },
    )
    _json(run / "manifest.json", {"status": "COMPLETE", "freeze_sha256": freeze_hash})
    for sequence in SEQUENCES:
        path = run / "results" / sequence / "results.txt"
        path.parent.mkdir(parents=True)
        lines = ["\t".join(COLUMNS)] + [
            f"{item['index']}\t{item['timestamp_text']}\t-2.5\t0.125"
            for item in identities[sequence]
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _repackage(run)
    return run, data, clocks, rows, tmp_path


def _audit(fixture):
    run, data, clocks, _rows, project = fixture
    return audit_run(
        run, data, project_root=project, clock_reader=lambda folder: clocks[folder.name]
    )


def test_complete_signed_empty_observation_population_passes(complete_run):
    report = _audit(complete_run)
    assert report["status"] == "PASS"
    assert report["rows_verified"] == 434
    assert report["aggregate"]["negative_ttc_rows"] == 434
    assert report["aggregate"]["zero_roi_rows"] == 434
    assert report["aggregate"]["producer_diagnostics"]["event"]["unknown_count"] == 868
    assert report["max_complete_sensor_span_us"] == 300_000


@pytest.mark.parametrize(
    "column,replacement,message",
    [
        (1, "0.3", "timestamp text"),
        (2, "2.5", "differs from ledger"),
        (3, "0.001", "differs from ledger"),
    ],
)
def test_resealed_results_tampering_cannot_evade_ledger_and_csv_checks(
    complete_run, column, replacement, message
):
    run = complete_run[0]
    path = run / "results" / SEQUENCES[0] / "results.txt"
    lines = path.read_text().splitlines()
    cells = lines[1].split("\t")
    cells[column] = replacement
    lines[1] = "\t".join(cells)
    path.write_text("\n".join(lines) + "\n")
    _repackage(run)
    with pytest.raises(AuditError, match=message):
        _audit(complete_run)


def test_duplicate_zip_member_is_rejected_even_with_updated_archive_hash(complete_run):
    run = complete_run[0]
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(run / "results.zip", "a") as archive:
            archive.writestr(f"{SEQUENCES[0]}/results.txt", "duplicate")
    for name in ("summary.json", "manifest.json"):
        value = json.loads((run / name).read_text())
        value["results_zip_sha256"] = file_sha(run / "results.zip")
        _json(run / name, value)
    with pytest.raises(AuditError, match="duplicate members"):
        _audit(complete_run)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("past_coverage", "crosses recorded coverage"),
        ("future_bbox", "Future or wrong bbox"),
        ("wrong_duration", "delta differs"),
        ("full_span", "exceeds 650ms"),
    ],
)
def test_temporal_evidence_rejects_causal_and_duration_violations(complete_run, mutation, message):
    _run, _data, clocks, rows, _project = complete_run
    row = copy.deepcopy(rows[0])
    identity = row["identity"]
    observation = row["input_details"]["event_observations"][0]
    if mutation == "past_coverage":
        observation["windows_us"][0][0] = -1
    elif mutation == "future_bbox":
        row["input_details"]["bbox_sources"][0]["anchor_us"] += 1
    elif mutation == "wrong_duration":
        observation["delta_t_s"][0] = 0.08
    else:
        observation["first_sensor_ts_us"] = identity["anchor_us"] - 650_001
    with pytest.raises(AuditError, match=message):
        validate_temporal(row, clocks[SEQUENCES[0]], [identity])


def test_modified_raw_file_is_detected(complete_run):
    path = complete_run[1] / SEQUENCES[0] / "data.hdf5"
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    with pytest.raises(AuditError, match="SHA-256 changed"):
        _audit(complete_run)


def test_future_rgb_frame_is_rejected_even_if_its_raw_clock_is_real(complete_run):
    _run, _data, clocks, rows, _project = complete_run
    row = copy.deepcopy(rows[0])
    query = row["identity"]["anchor_us"]
    row["input_details"]["rgb_observations"] = [
        {
            "timestamps_us": [query, query + 100_000],
            "frame_indices": [0, 1],
            "first_sensor_ts_us": query,
            "anchor_us": query + 100_000,
            "available_us": query + 100_000,
            "delta_t_s": [0.1],
        }
    ]
    with pytest.raises(AuditError, match="Future or reversed observation"):
        validate_temporal(row, clocks[SEQUENCES[0]], [row["identity"]])


def test_partial_run_is_rejected_before_any_sensor_access(complete_run):
    run, data, *_ = complete_run
    summary = json.loads((run / "summary.json").read_text())
    summary["status"] = "PARTIAL"
    _json(run / "summary.json", summary)
    with pytest.raises(AuditError, match="not terminal"):
        audit_run(run, data, clock_reader=lambda _folder: pytest.fail("read partial run sensors"))


def _unrepresentable_diagnostics(observation, *, value=0.05, known=False):
    classification = (
        "unrepresentable_nan"
        if value == "nan"
        else "unrepresentable_zero"
        if value == 0
        else "unrepresentable_positive_le_model_delta"
    )
    output = {
        "producer_id": "E_A5_MATCHED",
        "ttc_seconds": [value],
        "domain_classification": [classification],
        "unrepresentable": True,
        "known_mask": [known],
        "sensor_support": [[0.0, 0.0, 0.0]],
    }
    return {
        "known_true": int(known),
        "known_total": 1,
        "support_count": 3,
        "support_min": 0.0,
        "support_max": 0.0,
        "observations_requested": 1,
        "observations_accepted": 0,
        "accepted_observation_indices": [],
        "accepted_anchor_us": [],
        "observations_rejected_unrepresentable": 1,
        "rejected_compute_seconds": 0.02,
        "rejected": [
            {
                "observation_index": 0,
                "anchor_us": observation["anchor_us"],
                "reason": "unrepresentable_expert_ttc",
                "compute_seconds": 0.02,
                "invalid_producers": [copy.deepcopy(output)],
                "all_producer_outputs": [copy.deepcopy(output)],
            }
        ],
    }


def _empty_diagnostics():
    return {
        "known_true": 0,
        "known_total": 0,
        "support_count": 0,
        "observations_requested": 0,
        "observations_accepted": 0,
        "accepted_observation_indices": [],
        "accepted_anchor_us": [],
        "observations_rejected_unrepresentable": 0,
        "rejected_compute_seconds": 0.0,
        "rejected": [],
    }


@pytest.fixture
def terminal_partial(complete_run):
    run, _data, _clocks, rows, _project = complete_run
    row = rows[0]
    observation = row["input_details"]["event_observations"][0]
    row.update(
        status="unavailable",
        prediction_ttc_seconds=None,
        reason="event_observations_unrepresentable",
    )
    row["prediction"].update(
        status="unavailable",
        model_ttc=None,
        event_ttc=None,
        rgb_ttc=None,
        model_reason=row["reason"],
        event_history_count=0,
        producer_diagnostics={
            "event": _unrepresentable_diagnostics(observation),
            "rgb": _empty_diagnostics(),
        },
    )
    row["runtime_internal"] = copy.deepcopy(row["prediction"])
    (run / "all_rows.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows))
    partial_root = run / "partial_results"
    entries = []
    for sequence in SEQUENCES:
        old = run / "results" / sequence / "results.txt"
        lines = old.read_text().splitlines()
        if sequence == SEQUENCES[0]:
            del lines[1]
        new = partial_root / sequence / "available_predictions.tsv"
        new.parent.mkdir(parents=True)
        new.write_text("\n".join(lines) + "\n")
        entries.append(
            {
                "path": new.relative_to(partial_root).as_posix(),
                "sequence_id": sequence,
                "available_rows": len(lines) - 1,
                "sha256": file_sha(new),
            }
        )
        old.unlink()
        old.parent.rmdir()
    (run / "results").rmdir()
    (run / "results.zip").unlink()
    identity = row["identity"]
    unavailable = {
        key: identity[key] for key in ("sequence_id", "row_index", "index", "timestamp_text")
    }
    unavailable.update(status="unavailable", reason=row["reason"])
    _json(
        partial_root / "partial_manifest.json",
        {
            "status": "PARTIAL_NOT_SUBMISSION",
            "expected_rows": 434,
            "available_rows": 433,
            "unavailable_rows": [unavailable],
            "sequence_files": entries,
        },
    )
    (partial_root / "PARTIAL_NOT_SUBMISSION.txt").write_text("PARTIAL RESULTS ONLY\n")
    with zipfile.ZipFile(run / "partial_results.zip", "w") as archive:
        for path in partial_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(partial_root).as_posix())
    summary = json.loads((run / "summary.json").read_text())
    summary.update(
        status="PARTIAL",
        ok_rows=433,
        null_rows=1,
        terminal_population_accounted=True,
        prepare_only=False,
        limit=None,
        available_prediction_rows=433,
        partial_results="partial_results",
        partial_results_zip="partial_results.zip",
        partial_available_rows=433,
        partial_unavailable_rows=1,
        partial_results_zip_sha256=file_sha(run / "partial_results.zip"),
        producer_known_mask_diagnostics={
            "event": {"known_true": 0, "known_total": 867, "support_count": 2601},
            "rgb": {"known_true": 0, "known_total": 0, "support_count": 0},
        },
    )
    _json(run / "summary.json", summary)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest.update(
        status="PARTIAL", partial_results_zip_sha256=summary["partial_results_zip_sha256"]
    )
    _json(run / "manifest.json", manifest)
    return complete_run


def test_terminal_partial_preserves_null_and_cannot_claim_complete_submission(terminal_partial):
    report = _audit(terminal_partial)
    assert report["audit_status"] == "PASS"
    assert report["status"] == "PARTIAL"
    assert report["package"]["submission_complete"] is False
    assert report["package"]["available_rows"] == 433
    assert report["aggregate"]["unavailable_prediction_rows"] == 1
    assert (
        report["aggregate"]["producer_diagnostics"]["event"][
            "observations_rejected_unrepresentable"
        ]
        == 1
    )


def test_partial_cannot_retain_complete_named_zip(terminal_partial):
    (terminal_partial[0] / "results.zip").write_bytes(b"incorrect complete artifact")
    with pytest.raises(AuditError, match="PARTIAL run must not contain"):
        _audit(terminal_partial)


def test_known_mask_does_not_decide_phase_domain_rejection(terminal_partial):
    row = copy.deepcopy(terminal_partial[3][0])
    events = row["input_details"]["event_observations"]
    # A true known mask cannot make a mathematically invalid 0.05s representable.
    row["prediction"]["producer_diagnostics"]["event"] = _unrepresentable_diagnostics(
        events[0], known=True
    )
    validate_admission(row, events, [])
    # Conversely, unknown does not justify rejecting a valid negative signed TTC.
    diagnostic = _unrepresentable_diagnostics(events[0], value=-2.5, known=False)
    row["prediction"]["producer_diagnostics"]["event"] = diagnostic
    with pytest.raises(AuditError, match="domain classification"):
        validate_admission(row, events, [])


def test_explicit_nan_diagnostic_string_is_not_a_fabricated_finite_prediction(terminal_partial):
    row = copy.deepcopy(terminal_partial[3][0])
    events = row["input_details"]["event_observations"]
    row["prediction"]["producer_diagnostics"]["event"] = _unrepresentable_diagnostics(
        events[0], value="nan"
    )
    validate_admission(row, events, [])
    assert row["prediction_ttc_seconds"] is None


def test_accepted_anchor_complement_must_stay_aligned(terminal_partial):
    row = copy.deepcopy(terminal_partial[3][0])
    row["prediction"]["producer_diagnostics"]["event"]["accepted_anchor_us"] = [123]
    with pytest.raises(AuditError, match="Accepted anchors"):
        validate_admission(row, row["input_details"]["event_observations"], [])


def test_v2_mixed_prefix_history_requires_all_predeclared_anchors():
    query = 350_000
    plans = []
    for anchor in (50_000, 150_000, 250_000, 350_000):
        first = max(0, anchor - 300_000)
        edges = [first + (anchor - first) * index // 3 for index in range(4)]
        plans.append({"windows_us": [[edges[i], edges[i + 1]] for i in range(3)]})
    row = {"identity": {"anchor_us": query}, "input_details": {"event_observations": plans}}
    clock = {"event_start_us": 0, "event_end_us": 1_000_000}
    validate_v2_plan(row, clock)
    plans.pop(1)
    with pytest.raises(AuditError, match="frozen v2 real-prefix"):
        validate_v2_plan(row, clock)


def test_nonfinite_json_is_not_accepted(complete_run):
    run = complete_run[0]
    path = run / "all_rows.jsonl"
    path.write_text(
        path.read_text().replace(
            '"prediction_ttc_seconds": -2.5', '"prediction_ttc_seconds": NaN', 1
        )
    )
    with pytest.raises(AuditError, match="Non-finite JSON"):
        _audit(complete_run)


def test_independent_ros_clock_reader_uses_payload_not_bag_clock(tmp_path):
    def fields(values):
        return b"".join(
            struct.pack("<I", len(key) + 1 + len(value)) + key + b"=" + value
            for key, value in values.items()
        )

    def record(header, payload):
        encoded = fields(header)
        return struct.pack("<I", len(encoded)) + encoded + struct.pack("<I", len(payload)) + payload

    def message(connection, payload):
        return record(
            {
                b"op": b"\x02",
                b"conn": struct.pack("<I", connection),
                b"time": struct.pack("<II", 9999, 0),
            },
            payload,
        )

    def header(seconds, nanoseconds=0):
        return struct.pack("<IIII", 0, seconds, nanoseconds, 0)

    # Payload origin is 10 seconds; bag playback time deliberately differs.
    events = header(10, 900_000) + struct.pack("<III", 480, 640, 2)
    events += struct.pack("<HHIIB", 1, 1, 10, 500_000, 1)
    events += struct.pack("<HHIIB", 2, 2, 10, 1_999_999, 0)
    empty = header(0) + struct.pack("<III", 480, 640, 0)
    chunk = message(1, empty) + message(2, header(10)) + message(1, events)
    chunk += message(2, header(10, 2_000_000))
    bag = b"#ROSBAG V2.0\n" + record({b"op": b"\x05", b"compression": b"none"}, chunk)
    for connection, topic, kind in (
        (1, b"/capture_node/events", b"dv_ros_msgs/EventArray"),
        (2, b"/image_raw", b"sensor_msgs/Image"),
    ):
        bag += record(
            {b"op": b"\x07", b"conn": struct.pack("<I", connection), b"topic": topic},
            fields({b"type": kind}),
        )
    (tmp_path / "raw.bag").write_bytes(bag)
    clocks = sensor_clocks(tmp_path)
    assert clocks["rgb_us"] == [0, 2000]
    assert clocks["event_start_us"] == 500
    assert clocks["event_end_us"] == 2000
    assert clocks["clock_origin_ns"] == 10_000_000_000
