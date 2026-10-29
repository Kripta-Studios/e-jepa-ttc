from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.data.eap_highres_jepa import (
    BlockAwareBatchSampler,
    EAPHighResLabelFreeDataset,
    collate_label_free,
)
from e_jepa_ttc.data.matched_eap_subset import (
    ALLOWED_PARQUET_COLUMNS,
    LABEL_FAMILY_PROVENANCE,
    MatchedSubsetConfig,
    build_matched_eap_subset,
    sha256_json,
)


class _Reader:
    opened = 0
    closed = 0

    def __init__(self, path: Path) -> None:
        self.path = path

    def open(self) -> None:
        type(self).opened += 1

    def close(self) -> None:
        type(self).closed += 1

    def read_window(self, start_us: int, end_us: int) -> dict[str, np.ndarray]:
        return {
            "x": np.asarray([0, 4], dtype=np.int32),
            "y": np.asarray([0, 4], dtype=np.int32),
            "t": np.asarray([start_us, end_us - 1], dtype=np.int64),
            "p": np.asarray([1, -1], dtype=np.int8),
        }


def _manifest() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    horizons = (0.1, 0.2, 0.3)
    for index in range(4):
        timestamp = 1_000_000 + index * 100_000
        endpoints = []
        endpoint_ids: dict[str, str] = {}
        endpoint_timestamps: dict[str, int] = {}
        for horizon_index, horizon in enumerate(horizons):
            endpoint_timestamp = timestamp + int(horizon * 1_000_000)
            endpoint_id = f"endpoint-{index}-{horizon_index}"
            endpoint_ids[str(horizon)] = endpoint_id
            endpoint_timestamps[str(horizon)] = endpoint_timestamp
            endpoints.append(
                {
                    "horizon_s": horizon,
                    "row_id": endpoint_id,
                    "sample_token": endpoint_id,
                    "timestamp_us": endpoint_timestamp,
                    "events_path": "events.h5",
                    "event_windows_us": [
                        [endpoint_timestamp - 200_000, endpoint_timestamp - 100_000],
                        [endpoint_timestamp - 100_000, endpoint_timestamp],
                    ],
                    "sequence_id": "sequence",
                    "track_id": "track",
                }
            )
        row_id = chr(ord("a") + index)
        rows.append(
            {
                "row_id": row_id,
                "sequence_id": "sequence",
                "track_id": "track",
                "sample_token": f"sample-{index}",
                "timestamp_us": timestamp,
                "events_path": "events.h5",
                "event_windows_us": [
                    [timestamp - 200_000, timestamp - 100_000],
                    [timestamp - 100_000, timestamp],
                ],
                "role": "train",
                "block_id": "block",
                "endpoint_row_ids": endpoint_ids,
                "endpoint_timestamps_us": endpoint_timestamps,
                "future_endpoints": endpoints,
                "candidate_row_ids": ["a", "b", "c", "d"],
            }
        )
    manifest: dict[str, object] = {
        "artifact_type": "matched_eap_subset_v1",
        "schema_version": "matched_eap_subset_v1",
        "code_commit": "a" * 40,
        "source": {
            "projected_columns": list(ALLOWED_PARQUET_COLUMNS),
            "parquet_sha256": "source",
        },
        "label_family_provenance": dict(LABEL_FAMILY_PROVENANCE),
        "config": {
            "horizons_s": list(horizons),
            "horizon_tolerance_s": 0.025,
            "exclusion_window_s": 0.02,
            "minimum_anchors_per_block": 4,
            "max_anchors_per_block": 4,
            "minimum_negatives": 2,
            "stage_sizes": [4],
            "seed": 7,
            "temporal_steps": 5,
            "ssl_width": 320,
            "ssl_height": 192,
            "bins": 5,
            "batch_size": 2,
            "max_workers": 8,
            "update_budget": 1000,
            "calibration_mode": "focal",
            "signed_ttc_convention": "signed_seconds_future_minus_anchor",
        },
        "split_assignments": {"train": ["sequence"], "validation": ["heldout"]},
        "split_hash": "split",
        "selection_rule": "label_free_fixed_four_anchor_blocks_round_robin_v2",
        "sampler_order": ["a", "b", "c", "d"],
        "sampler_order_hash": sha256_json(["a", "b", "c", "d"]),
        "freeze": {
            "modalities": ["events"],
            "ssl_input_policy": "full_frame_event_only_320x192_from_raw",
            "downstream_input_policy": "official_square_object_roi_128x128_post_ssl_only",
            "temporal_steps": 5,
            "calibration_mode": "focal",
            "signed_ttc_convention": "signed_seconds_future_minus_anchor",
            "seeds": [7, 13, 23],
            "batch_size": 2,
            "max_workers": 8,
            "update_budget": 1000,
        },
        "selection_report": {
            "unavailable_stages": [],
            "nce_anchor_coverage": 1.0,
            "minimum_negatives": 2,
            "nce_by_stage": {
                "matched_4": {
                    "overall": {
                        "gate_passed": True,
                        "valid_anchor_fraction": 1.0,
                        "minimum_negatives": 2,
                    },
                    "by_role": {
                        "train": {
                            "gate_passed": True,
                            "valid_anchor_fraction": 1.0,
                            "minimum_negatives": 2,
                        },
                        "validation": {
                            "gate_passed": False,
                            "valid_anchor_fraction": 0.0,
                            "minimum_negatives": 0,
                        },
                    },
                }
            },
        },
        "stages": [
            {
                "stage": "matched_4",
                "nominal_row_count": 4,
                "actual_row_count": 4,
                "block_ids": ["block"],
                "row_ids": ["a", "b", "c", "d"],
            }
        ],
        "blocks": [
            {
                "block_id": "block",
                "role": "train",
                "sequence_id": "sequence",
                "track_id": "track",
                "anchor_count": 4,
                "row_ids": ["a", "b", "c", "d"],
                "nce": {"gate_passed": True, "valid_anchor_fraction": 1.0, "minimum_negatives": 2},
            }
        ],
        "rows": rows,
        "dataset_hashes": {
            "source_parquet_sha256": "source",
            "config_sha256": "config",
            "split_sha256": "split",
            "sampler_order_sha256": sha256_json(["a", "b", "c", "d"]),
        },
    }
    manifest["matched_manifest_hash"] = sha256_json(
        {
            "code_commit": manifest["code_commit"],
            "rows": rows,
            "stages": manifest["stages"],
            "config": manifest["config"],
            "split_hash": manifest["split_hash"],
            "source_parquet_sha256": manifest["source"]["parquet_sha256"],
            "sampler_order_hash": manifest["sampler_order_hash"],
        }
    )
    manifest["artifact_sha256"] = manifest["matched_manifest_hash"]
    manifest["signature"] = sha256_json(manifest)
    return manifest


def test_shapes_endpoints_and_close(tmp_path: Path) -> None:
    (tmp_path / "events.h5").write_bytes(b"fixture")
    dataset = EAPHighResLabelFreeDataset(
        _manifest(), eap_root=tmp_path, role="train", stage="matched_4", reader_factory=_Reader
    )
    sample = dataset[0]
    assert sample.context_events.shape == (1, 5, 21, 192, 320)
    assert sample.future_events.shape == (1, 3, 5, 21, 192, 320)
    assert sample.horizon_delta_t_s.dtype == torch.float64
    assert sample.future_timestamps_s.dtype == torch.float64
    assert sample.nce_candidate_mask is not None
    assert sample.nce_candidate_mask.shape == (1, 3, 1, 3)
    batch = collate_label_free([sample, sample])
    assert batch.nce_candidate_mask is not None
    assert batch.nce_candidate_mask.shape == (2, 3, 2, 3)
    shifted = replace(
        sample,
        horizon_delta_t_s=sample.horizon_delta_t_s + 0.005,
        future_timestamps_s=sample.future_timestamps_s + 0.005,
    )
    centered = collate_label_free([sample, shifted], exclusion_window_s=0.02)
    assert centered.nce_candidate_mask is not None
    # The shifted candidate is close to the desired future (1.100s), not to
    # the anchor reference (1.000s), and must therefore be excluded.
    assert not bool(centered.nce_candidate_mask[0, 0, 1, 0])
    dataset.close()
    assert _Reader.closed >= 1


def test_role_filter_keeps_rows_and_block_ids_aligned_for_interleaved_blocks(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").mkdir()
    source_rows: list[dict[str, object]] = []
    for sequence, role_offset in (("train-sequence", 0), ("validation-sequence", 1)):
        for index in range(8):
            timestamp = (index + role_offset * 20) * 100_000
            source_rows.append(
                {
                    "sequence_id": sequence,
                    "sample_token": f"{sequence}-{index}",
                    "track_id": f"track-{role_offset}",
                    "public_track_id": f"track-{role_offset}",
                    "timestamp_us": timestamp,
                    "frame_timestamps_us": [[timestamp - 1, timestamp]],
                    "events_path": f"events-{role_offset}.h5",
                    "event_windows_us": [
                        [timestamp - 200_000, timestamp - 100_000],
                        [timestamp - 100_000, timestamp],
                    ],
                }
            )
    pd.DataFrame(source_rows).to_parquet(tmp_path / "data" / "train.parquet")
    split_path = tmp_path / "split.json"
    split_path.write_text(
        '{"assignments":{"train":["train-sequence"],"validation":["validation-sequence"]}}',
        encoding="utf-8",
    )
    manifest = build_matched_eap_subset(
        tmp_path,
        split_path,
        config=MatchedSubsetConfig(stage_sizes=(8,)),
        code_commit="a" * 40,
    )
    train = EAPHighResLabelFreeDataset(
        manifest,
        eap_root=tmp_path,
        role="train",
        stage="matched_8",
        reader_factory=_Reader,
    )
    validation = EAPHighResLabelFreeDataset(
        manifest,
        eap_root=tmp_path,
        role="validation",
        stage="matched_8",
        reader_factory=_Reader,
    )
    assert len(train.rows) == len(train.block_ids)
    assert len(validation.rows) == len(validation.block_ids)
    assert all(row["role"] == "train" for row in train.rows)
    assert all(row["role"] == "validation" for row in validation.rows)
    for batch_indices in BlockAwareBatchSampler(train, batch_size=2):
        assert len(batch_indices) <= 2
        assert len({train.block_ids[index] for index in batch_indices}) == 1
        assert len({train.rows[index]["track_id"] for index in batch_indices}) == 1
    train.close()
    validation.close()
