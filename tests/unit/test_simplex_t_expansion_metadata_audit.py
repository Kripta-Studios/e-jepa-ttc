"""Expansion metadata auditing uses explicit inputs and rejects mismatched interfaces."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def expansion_fixture(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/audit_simplex_t_expansion_metadata.py"
    spec = importlib.util.spec_from_file_location("expansion_audit_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "EXPECTED_QUERIES", 22)
    for directory in (
        "data_roles",
        "expansion_raw_bindings",
        "expansion_usability",
        "expansion_inventory",
    ):
        (tmp_path / directory).mkdir()
    sequences = [f"s{i:02}" for i in range(22)]
    roles = dict(expansion=sequences, original=["old"], protected=["p"], confirmation=["c"])
    (tmp_path / "data_roles/DATA_ROLES.json").write_text(json.dumps(dict(roles=roles)))
    rows, bindings, sources = [], [], {}
    for sequence in sequences:
        raw = tmp_path / sequence
        raw.write_bytes(b"fixture")
        stamp = raw.stat()
        sources[sequence] = dict(
            path=str(raw), bytes=stamp.st_size, mtime_ns=stamp.st_mtime_ns, sha256="f" * 64
        )
        rows.append(
            dict(
                sample_token=sequence,
                sequence_id=sequence,
                track_id="track",
                timestamp_us=200,
                frame_timestamps_us=[100, 200],
                event_windows_us=[[0, 100], [100, 200]],
                boxes_xyxy=[[0, 0, 8, 8], [0, 0, 8, 8]],
                events_path=f"{sequence}/events.h5",
                ttc=999.0,
            )
        )
        for window in range(2):
            bindings.append(
                dict(
                    sample_token=sequence,
                    sequence_id=sequence,
                    track_id="track",
                    window_id=window,
                    window_start_us=window * 100,
                    window_end_us=(window + 1) * 100,
                    target_anchor_event_clock_us=200,
                    frame_to_event_clock_offset_us=0,
                    events_path_relative=f"{sequence}/events.h5",
                    h5_file_sha256="f" * 64,
                    roi_x0=-2.0,
                    roi_y0=-2.0,
                    roi_x1=10.0,
                    roi_y1=10.0,
                    start_event_index=0,
                    stop_event_index=3,
                    full_window_event_count=3,
                    time_unit="us",
                )
            )
    pd.DataFrame(rows).to_parquet(tmp_path / "expansion_usability/USABLE_METADATA.parquet")
    pd.DataFrame(rows).to_csv(tmp_path / "expansion_inventory/SELECTED_METADATA.csv", index=False)
    pd.DataFrame(bindings).to_csv(tmp_path / "expansion_raw_bindings/RAW_BINDINGS.csv", index=False)
    (tmp_path / "expansion_raw_bindings/RAW_BINDING_MANIFEST.json").write_text(
        json.dumps(dict(identity=dict(sources=sources)))
    )

    def repin():
        monkeypatch.setattr(
            module,
            "PINS",
            {
                name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                for name in module.PINS
            },
        )

    repin()
    return module, tmp_path, repin


def test_input_only_expansion_audit(expansion_fixture):
    module, root, _ = expansion_fixture
    result = module.audit(root)
    assert result["queries"] == 22 and result["windows"] == 44
    assert result["prospective_selected_rows_equal_usable_rows"]
    assert "ttc" not in result["metadata_columns_read"]
    assert result["optimizer_updates"] == 0 and not result["raw_media_read"]


@pytest.mark.parametrize("failure", ["hash", "role", "window", "selection"])
def test_expansion_interface_corruption_is_rejected(expansion_fixture, failure):
    module, root, repin = expansion_fixture
    if failure in {"hash", "role"}:
        path = root / "data_roles/DATA_ROLES.json"
        roles = json.loads(path.read_text())
        roles["roles"]["protected"].append("s00")
        path.write_text(json.dumps(roles))
    else:
        path = root / (
            "expansion_raw_bindings/RAW_BINDINGS.csv"
            if failure == "window"
            else "expansion_inventory/SELECTED_METADATA.csv"
        )
        table = pd.read_csv(path)
        if failure == "window":
            table.loc[0, "window_end_us"] = 99
        else:
            table.loc[0, "sample_token"] = "unregistered"
        table.to_csv(path, index=False)
    if failure != "hash":
        repin()
    with pytest.raises(ValueError):
        module.audit(root)
