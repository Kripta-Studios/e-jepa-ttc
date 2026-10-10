from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from e_jepa_ttc.rgb_port.contracts import RGBFrame, assign_roles, assignment_sha256
from e_jepa_ttc.rgb_port.data import (
    TarFrameReader,
    collate_rgb_port,
    decode_query,
    make_rgb_producer_source,
)
from e_jepa_ttc.rgb_port.history import observation_triplets, producer_frames, select_frames
from e_jepa_ttc.rgb_port.normalization import (
    RGB_PHASE17_SHA256,
    FrozenNormalizer,
    fit_normalizer,
)


def _frame(index: int) -> RGBFrame:
    return RGBFrame(
        frame_id=f"f{index}",
        timestamp_us=index * 100_000,
        available_us=index * 100_000,
        shard_path="frames.tar",
        member_path=f"rgb/000000000_{index * 100_000:09d}.png",
        box_xyxy=(4.0 + index, 5.0, 15.0 + index, 18.0),
    )


def test_roles_are_deterministic_disjoint_24_8_8() -> None:
    groups = [f"sequence-{index:02d}" for index in range(40)]
    first = assign_roles(groups)
    second = assign_roles(list(reversed(groups)))
    assert first == second
    assert {role: list(first.values()).count(role) for role in ("P", "H", "V")} == {
        "P": 24,
        "H": 8,
        "V": 8,
    }
    assert assignment_sha256(first) == assignment_sha256(second)


def test_history_is_causal_distinct_and_retains_real_t2() -> None:
    frames = [_frame(index) for index in range(10)]
    assert [item.frame_id for item in producer_frames(frames[:2], cutoff_us=100_000)] == [
        "f0",
        "f1",
    ]
    selected = select_frames(frames, cutoff_us=900_000)
    assert len(selected) == 7  # 650 ms lookback at a 100 ms cadence
    observations = observation_triplets(frames, cutoff_us=900_000)
    assert len(observations) == 5
    assert all(len({frame.frame_id for frame in triplet}) == 3 for triplet in observations)
    assert max(frame.timestamp_us for triplet in observations for frame in triplet) <= 900_000


def _tar_with_frames(path: Path) -> None:
    with tarfile.open(path, "w") as archive:
        for index in range(2):
            array = np.full((24, 30, 3), 30 + index * 100, dtype=np.uint8)
            output = io.BytesIO()
            Image.fromarray(array).save(output, format="PNG")
            payload = output.getvalue()
            info = tarfile.TarInfo(f"rgb/000000000_{index * 100_000:09d}.png")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_real_tar_decode_preserves_uint8_and_unit_rgb(tmp_path: Path) -> None:
    _tar_with_frames(tmp_path / "frames.tar")
    row = {
        "producer_shards": ["frames.tar", "frames.tar"],
        "producer_members": ["rgb/000000000_000000000.png", "rgb/000000000_000100000.png"],
        "producer_times_us": [0, 100_000],
        "producer_sensor_times_us": [5_000_000, 5_100_058],
        "producer_boxes_xyxy": [[4, 5, 15, 18], [5, 5, 16, 18]],
        "roi_xyxy": [4, 4, 17, 17],
    }
    reader = TarFrameReader(tmp_path)
    try:
        decoded = decode_query(row, eap_root=tmp_path, reader=reader)
    finally:
        reader.close()
    assert decoded["rgb_uint8"].shape == (2, 3, 128, 128)
    assert decoded["rgb_uint8"].dtype == np.uint8
    assert decoded["rgb"].dtype == np.float32
    np.testing.assert_allclose(decoded["rgb"] * 255.0, decoded["rgb_uint8"], atol=1e-5)
    assert decoded["foreground_mask"].shape == (2, 1, 128, 128)
    assert decoded["boxes_in_crop_xyxy"].shape == (2, 4)
    assert decoded["delta_t_s"].item() == pytest.approx(0.100058)
    assert decoded["annotation_frame_times_us"].tolist() == [0, 100_000]
    assert decoded["sensor_frame_times_us"].tolist() == [5_000_000, 5_100_058]


def test_tar_reader_reuses_immutable_decoded_frames_with_bounded_lru(tmp_path: Path) -> None:
    _tar_with_frames(tmp_path / "frames.tar")
    reader = TarFrameReader(tmp_path, max_frame_cache_bytes=24 * 30 * 3 + 1)
    try:
        first = reader.read_uint8("frames.tar", "rgb/000000000_000000000.png")
        repeated = reader.read_uint8("frames.tar", "rgb/000000000_000000000.png")
        assert first is repeated
        assert not first.flags.writeable
        reader.read_uint8("frames.tar", "rgb/000000000_000100000.png")
        assert reader._frame_cache_bytes <= reader.max_frame_cache_bytes
    finally:
        reader.close()


def test_collate_rejects_mixed_real_history_lengths() -> None:
    base = {
        "rgb": np.zeros((2, 3, 2, 2), np.float32),
        "rgb_uint8": np.zeros((2, 3, 2, 2), np.uint8),
    }
    other = {**base, "rgb": np.zeros((3, 3, 2, 2), np.float32)}
    with pytest.raises(ValueError, match="separate sampler buckets"):
        collate_rgb_port([base, other])


def test_producer_source_factory_is_bound_to_raw_uint8_dino_sidecar(tmp_path: Path) -> None:
    import hashlib

    _tar_with_frames(tmp_path / "frames.tar")
    rows = pd.DataFrame(
        [
            {
                "sample_token": "token",
                "sequence_id": "sequence",
                "track_id": "track",
                "group_id": "sequence",
                "producer_shards": ["frames.tar", "frames.tar"],
                "producer_members": [
                    "rgb/000000000_000000000.png",
                    "rgb/000000000_000100000.png",
                ],
                "producer_times_us": [0, 100_000],
                "producer_sensor_times_us": [5_000_000, 5_100_058],
                "producer_boxes_xyxy": [[4, 5, 15, 18], [5, 5, 16, 18]],
                "roi_xyxy": [4, 4, 17, 17],
                "target_ttc": -2.0,
                "target_phase": -0.04879,
                "mass": 1.0,
                "query_time_us": 100_000,
                "rgb_anchor_us": 100_000,
                "query_delay_us": 0,
                "ordinal": 0,
            }
        ]
    )
    rows_path = tmp_path / "P_ROWS.parquet"
    rows.to_parquet(rows_path, index=False)
    manifest_path = tmp_path / "P_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "role": "P",
                "eap_root": str(tmp_path),
                "rows_path": str(rows_path),
                "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
                "split_assignment_sha256": "split",
            }
        ),
        encoding="utf-8",
    )
    targets_path = tmp_path / "P_DINO_TARGETS.npz"
    np.savez_compressed(
        targets_path,
        sample_token=np.asarray(["token"]),
        relation_targets=np.zeros((1, 2, 6, 32, 32), dtype=np.float32),
        relation_valid=np.ones((1, 2, 6, 32, 32), dtype=np.bool_),
    )
    (tmp_path / "P_DINO_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "role_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "input_storage": "raw_uint8_rgb_no_double_divide",
                "targets_path": str(targets_path),
                "targets_sha256": hashlib.sha256(targets_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    source = make_rgb_producer_source(manifest_path)
    batch = source.batch([0], "rgb")
    assert source.population_size == 1
    assert source.frame_counts.tolist() == [2]
    assert batch.events.shape == (1, 2, 3, 128, 128)
    assert batch.delta_t_s.shape == (1, 1)
    assert batch.dinov3_relation_targets is not None
    assert source.identity["dino_input_storage"] == "raw_uint8_rgb_no_double_divide"


def test_normalizer_is_unique_observation_role_bound_and_loadable(tmp_path: Path) -> None:
    values = np.vstack((np.ones(17), np.ones(17) * 99, np.ones(17) * 3))
    normalizer = fit_normalizer(
        values,
        ["a", "a", "b"],
        fit_role="H",
        expected_role="H",
        producer_sha256="producer",
    )
    np.testing.assert_allclose(normalizer.mean, 2.0)
    path = tmp_path / "normalizer.json"
    normalizer.save(path)
    loaded = FrozenNormalizer.load(path)
    loaded.validate_endpoint(
        modality="rgb",
        fit_role="H",
        schema_sha256=RGB_PHASE17_SHA256,
        producer_sha256="producer",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        loaded.validate_endpoint(
            modality="event",
            fit_role="H",
            schema_sha256=RGB_PHASE17_SHA256,
            producer_sha256="producer",
        )
    with pytest.raises(ValueError, match="preregistered"):
        fit_normalizer(
            values[:1],
            ["v"],
            fit_role="V",
            expected_role="V",
            producer_sha256="producer",
        )


def test_input_contract_manifest_keeps_scoring_precision() -> None:
    path = Path("artifacts/rgb_port_20261008/INPUT_CONTRACTS.json")
    if not path.is_file():
        pytest.skip("real preparation artifact is not present")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"
    assert payload["rgb"]["range"] == [0.0, 1.0]
