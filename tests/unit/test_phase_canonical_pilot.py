"""Canonical-domain failures preserve whole-batch rejection and on-disk traces."""

from types import SimpleNamespace
from typing import cast

import numpy as np
from torch import nn

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.rgb_port.features import pair133
from operational.phase_timewarp.canonical_pair import CanonicalPair
from scripts import pilot_phase_canonical_pair as pilot


class FixedPair(nn.Module):
    """Emit a finite raw phase whose physical correction exceeds the domain."""

    def forward(self, batch: PairFeatureBatch):
        return batch.features.new_full((len(batch.features),), 2.0)


def test_domain_rejection_preserves_trace_and_original_reason(tmp_path, monkeypatch):
    runtime = SimpleNamespace(
        models={name: CanonicalPair(FixedPair()) for name in ("PAIR_E_MATCHED", "PAIR_R")}
    )
    calls = [
        {
            "call_id": 0,
            "prefix": "E",
            "observations": [
                {
                    "batch_index": 0,
                    "observation_id": "event:1000000",
                    "anchor_us": 1_000_000,
                    "available_us": 1_000_000,
                    "first_sensor_ts_us": 900_000,
                    "timestamps_us": [950_000, 1_000_000],
                    "windows_us": [[900_000, 950_000], [950_000, 1_000_000]],
                    "source_accepted": True,
                }
            ],
        }
    ]
    arrays = {
        "call0.pixels": np.ones((1, 2, 12, 128, 128), np.float32),
        "call0.delta": np.array([[0.05]], np.float32),
    }

    def expert(models, prefix, pixels, delta):
        assert prefix == "E"
        features = pair133(
            pixels.new_zeros((len(pixels), 128)),
            delta[:, -1],
            pixels.new_full((len(pixels), 2), 0.7),
        )
        models["PAIR_E_MATCHED"](PairFeatureBatch(features))
        raise AssertionError("Invalid corrected phase must reject the whole batch")

    saved = []

    def save_execution(destination, result, *args):
        saved.append(result)
        return None, {}

    monkeypatch.setattr(pilot.engine, "_expert_block", expert)
    monkeypatch.setattr(pilot, "save_execution", save_execution)
    monkeypatch.setattr(pilot, "native_vector", lambda *args: np.full(3, np.nan))
    result = pilot.execute(
        cast(pilot.FinalTransferPredictor, runtime),
        calls,
        arrays,
        1_000_000,
        1.0,
        "cpu",
        tmp_path,
    )

    assert len(saved) == 1
    assert result["accepted_ids"] == []
    record = result["calls"][0]
    assert record["status"] == "rejected" and record["domain_rejection"]
    assert "1-d0*q<=0" in record["canonical_domain_error"]
    assert "no singleton substitution" in record["rejection_scope"]
    with np.load(tmp_path / "CANONICAL_TRACES.npz", allow_pickle=False) as trace:
        key = "call0.PAIR_E_MATCHED."
        assert not trace[key + "returned"].item()
        assert trace[key + "domain_valid"].tolist() == [False]
        assert trace[key + "corrected_q"][0] > 10
        assert np.isnan(trace[key + "raw_corrected_phase"][0])
        assert trace[key + "input133"].shape == (1, 133)
