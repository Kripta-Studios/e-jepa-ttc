"""CPU-only tests for the concrete detected-PHASE panel adapter."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from operational.detection_phase_service.ttc_panel_adapter import (
    PanelAdapter,
    PanelAdapterConfig,
    PanelCadenceRuntime,
    load_config,
)
from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory


def _empty() -> dict[str, np.ndarray]:
    return {
        "x": np.empty(0, dtype=np.int32),
        "y": np.empty(0, dtype=np.int32),
        "t": np.empty(0, dtype=np.int64),
        "p": np.empty(0, dtype=np.int8),
    }


def test_config_validation_is_strict_and_does_not_construct_runtime(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    eap = tmp_path / "eap"
    candidate.mkdir()
    (eap / "data").mkdir(parents=True)
    (eap / "data/train.parquet").write_bytes(b"manifest")
    (eap / "events.h5").write_bytes(b"events")
    path = tmp_path / "adapter.json"
    path.write_text(
        json.dumps(
            {
                "schema": "detected_phase_panel_adapter_config_v1",
                "candidate_root": str(candidate),
                "eap_root": str(eap),
                "device": "cuda",
                "max_objects": 8,
                "logical_max_objects": 20,
                "rgb_workers": 2,
                "rgb_object_chunk_size": 1,
                "phase_bounds": True,
                "max_frame_cache_bytes": 0,
                "event_stream_sha256": {"events.h5": hashlib.sha256(b"events").hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    config, receipt = load_config(path)
    assert config.device == "cuda" and config.logical_max_objects == 20
    assert receipt["sha256"] and receipt["canonical_sha256"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["unknown"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fields differ"):
        load_config(path)


def test_cadence_runtime_buffers_unique_source_intervals_and_materializes_once() -> None:
    class History:
        def __init__(self) -> None:
            self.calls = []

        def initialize(self, raw, start, end) -> None:
            self.calls.append((start, end, len(raw["t"])))

    class Sealed:
        def __init__(self) -> None:
            self.calls = []
            self.closed = False
            self.last_end = None
            self.rebases = []

        def rebase_history(self, raw_history, start_us, end_us) -> None:
            self.rebases.append((start_us, end_us, len(raw_history["t"])))
            self.last_end = end_us

        def update(self, raw, start, end, objects, rgb):
            if start != self.last_end:
                raise ValueError("noncontiguous sealed update")
            self.calls.append((start, end, len(raw["t"]), len(objects)))
            self.last_end = end
            return {"objects": [{"object_id": "id"}], "packet_ingested_count": 1}

        def close(self) -> None:
            self.closed = True

    history, sealed, closed = History(), Sealed(), []
    runtime = PanelCadenceRuntime(sealed, history, max_objects=8, on_close=lambda: closed.append(1))
    intervals = [
        (80_000, 100_000),
        *[(value, value + 100_000) for value in range(100_000, 700_000, 100_000)],
    ]
    for index, (start, end) in enumerate(intervals):
        packet = _empty()
        if index != 2:  # A real empty source packet remains a valid, owned interval.
            packet["t"] = np.asarray([end - 1], dtype=np.int64)
            packet["x"] = np.asarray([1], dtype=np.int32)
            packet["y"] = np.asarray([1], dtype=np.int32)
            packet["p"] = np.asarray([1], dtype=np.int8)
        objects = (
            (ObjectRegion("id", (0.0, 0.0, 10.0, 10.0), None),)
            if index == len(intervals) - 1
            else ()
        )
        receipt = runtime.update(packet, start, end, objects, RawRGBHistory(end, {}, {}, ()))
        assert receipt["packet_ingested_count"] == 1
    assert history.calls == []
    assert sealed.rebases == [(80_000, 680_000, 5)]
    assert sealed.calls == [(680_000, 700_000, 1, 1)]
    assert receipt["source_interval_us"] == [600_000, 700_000]
    runtime.close()
    assert sealed.closed and closed == [1]


def test_frame_loader_reads_exact_contiguous_windows_and_preserves_empty_packet(
    tmp_path: Path,
) -> None:
    class Events:
        def __init__(self) -> None:
            self.windows = []

        def read_window(self, start, end):
            self.windows.append((start, end))
            if start == 100_000:
                return _empty()
            raw = _empty()
            raw.update(
                x=np.asarray([0], dtype=np.int32),
                y=np.asarray([0], dtype=np.int32),
                t=np.asarray([start], dtype=np.int64),
                p=np.asarray([1], dtype=np.int8),
            )
            return raw

        def close(self) -> None:
            pass

    class Frames:
        def read_uint8(self, shard, member):
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    root = tmp_path.resolve()
    config = PanelAdapterConfig(root, root, "cuda", 8, 20, 2, 1, True, 0, {"events.h5": "a" * 64})
    adapter = PanelAdapter(config, {})
    prefix = [
        {
            "sample_token": f"t{index}",
            "sequence_id": "seq",
            "events_path": "events.h5",
            "rgb_shard_path": "frames.tar",
            "rgb_member_path": f"{index}.png",
            "rgb_member_sha256": None,
            "exposure_end_us": end,
        }
        for index, end in enumerate((100_000, 200_000, 300_000))
    ]
    events = Events()
    adapter._active = {  # Deliberately inject CPU reader doubles; CUDA factory is not invoked.
        "condition": "R3",
        "query_id": "q",
        "prefix": prefix,
        "cursor": 0,
        "previous_end": None,
        "events": events,
        "frames": Frames(),
        "images": {},
        "timestamps": {},
    }
    query = {"query_id": "q", "sequence_id": "seq", "r3_label_free": {"prefix": prefix}}
    loaded = [adapter.frame_loader("R3", query, frame) for frame in prefix]
    assert events.windows == [(80_000, 100_000), (100_000, 200_000), (200_000, 300_000)]
    assert len(loaded[1][0]["t"]) == 0
    assert loaded[-1][2:] == (200_000, 300_000)
    assert loaded[-1][1].triplets == [(0, 1, 2)]
    with pytest.raises(ValueError, match="sealed prefix order"):
        adapter.frame_loader("R3", query, prefix[-1])


def test_runtime_rechecks_bound_sources_before_any_cuda_construction(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train.parquet").write_bytes(b"manifest")
    config = PanelAdapterConfig(
        tmp_path, tmp_path, "cuda", 8, 20, 2, 1, True, 0, {"events.h5": "a" * 64}
    )
    adapter = PanelAdapter(
        config,
        {
            "source_sha256": {"tampered": "0" * 64},
            "media_manifest_sha256": "not_reached",
        },
    )
    query = {
        "query_id": "q",
        "sequence_id": "seq",
        "r3_label_free": {
            "prefix": [
                {
                    "sequence_id": "seq",
                    "exposure_end_us": 700_000,
                    "events_path": "events.h5",
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="sources changed"):
        adapter.runtime_factory("R3", query)


def test_two_successive_irregular_mature_queries_use_public_history_rebase() -> None:
    class History:
        def initialize(self, raw, start_us, end_us) -> None:
            raise AssertionError("Adapter must use the sealed public rebase API")

    class Sealed:
        def __init__(self) -> None:
            self.last_end = None
            self.rebases = []
            self.calls = []

        def rebase_history(self, raw, start_us, end_us) -> None:
            self.last_end = end_us
            self.rebases.append((start_us, end_us))

        def update(self, raw, start_us, end_us, objects, rgb):
            if start_us != self.last_end:
                raise ValueError("noncontiguous sealed update")
            self.last_end = end_us
            self.calls.append((start_us, end_us))
            return {"objects": [{"object_id": "id"}], "packet_ingested_count": 1}

        def close(self) -> None:
            pass

    sealed = Sealed()
    runtime = PanelCadenceRuntime(sealed, History(), max_objects=8, on_close=lambda: None)
    intervals = [
        (80_000, 100_000),
        *[(value, value + 100_000) for value in range(100_000, 800_000, 100_000)],
    ]
    region = (ObjectRegion("id", (0.0, 0.0, 10.0, 10.0), None),)
    for index, (start, end) in enumerate(intervals):
        packet = _empty()
        objects = region if index >= len(intervals) - 2 else ()
        runtime.update(packet, start, end, objects, RawRGBHistory(end, {}, {}, ()))
    assert sealed.rebases == [(80_000, 680_000), (180_000, 780_000)]
    assert sealed.calls == [(680_000, 700_000), (780_000, 800_000)]


def test_prefix_media_preflight_hashes_rgb_and_events_before_runtime(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train.parquet").write_bytes(b"manifest")
    event = tmp_path / "events.h5"
    event.write_bytes(b"event-stream")
    archive = tmp_path / "frames.tar"
    payload = b"png-payload"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("frame.png")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    event_digest = hashlib.sha256(event.read_bytes()).hexdigest()
    config = PanelAdapterConfig(
        tmp_path, tmp_path, "cuda", 8, 20, 2, 1, True, 0, {"events.h5": event_digest}
    )
    adapter = PanelAdapter(config, {})
    prefix = [
        {
            "sample_token": "token",
            "events_path": "events.h5",
            "rgb_shard_path": "frames.tar",
            "rgb_member_path": "frame.png",
            "rgb_member_sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    receipt = adapter._preflight_prefix_media("R3", "query", prefix)
    assert receipt["event_stream_sha256"] == {"events.h5": event_digest}
    archive.write_bytes(b"changed")
    with pytest.raises((tarfile.TarError, ValueError)):
        adapter._preflight_prefix_media("R3", "query", prefix)
