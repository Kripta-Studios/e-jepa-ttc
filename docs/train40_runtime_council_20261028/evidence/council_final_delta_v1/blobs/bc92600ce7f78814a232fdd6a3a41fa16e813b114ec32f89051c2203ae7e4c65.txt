"""Fragment receipts follow exact lossless compression and never precede publication."""

import numpy as np

from operational.train40_system.prepare import configuration, write_shard


def test_historical_representation_and_train_only_scope_are_fixed():
    config = configuration()
    assert config.event_v4_storage_dtype == "float32"
    assert config.materialize_splits == ("train",)
    assert config.roi_size == 128
    assert config.event_v4_bins_per_polarity == 5
    assert config.event_pixel_diff == 5
    assert config.event_v4_margin_fraction == 0.25
    assert not config.include_rgb and not config.include_masks


def test_shard_preserves_every_float_and_declared_order(tmp_path, monkeypatch):
    import e_jepa_ttc.data.garlttc_lhr_cache as reference

    events = np.zeros((3, 12, 128, 128), dtype=np.float32)
    events[:, :, 17, 23] = np.nextafter(np.float32(1), np.float32(2))
    record = {
        "event_v4_common_roi": events,
        "sample_token": "token",
        "ttc_s": -2.5,
        "garl_visible_heights_px": np.array([1, 2], dtype=np.float32),
        "event_v4_boxes_xyxy": np.zeros((3, 4), dtype=np.float32),
        "event_v4_common_square_xyxy": np.zeros(4, dtype=np.float32),
        "observable_motion": np.zeros(10, dtype=np.float32),
        "garl_delta_t_s": 0.1,
    }
    monkeypatch.setattr(reference, "_materialize_row", lambda *args, **kwargs: record)
    raw = tmp_path / "data/train/sequence/events.h5"
    raw.parent.mkdir(parents=True)
    raw.touch()
    path = tmp_path / "shard.npz"
    receipt = write_shard(
        {
            "rows": [{"events_path": "data/train/sequence/events.h5"}],
            "ordinals": [42],
            "raw_root": str(tmp_path),
            "destination": str(path),
            "freeze_sha256": "freeze",
            "raw_sha256": {"sequence": "raw"},
        }
    )
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["events"][0], events)
        np.testing.assert_array_equal(data["ordinals"], [42])
        assert data["target_ttc"][0] == -2.5
    assert receipt["exact_compression_parity"]
    assert receipt["optimizer_updates"] == 0
    assert path.with_suffix(".json").is_file()
    assert not path.with_suffix(".pending.npz").exists()
