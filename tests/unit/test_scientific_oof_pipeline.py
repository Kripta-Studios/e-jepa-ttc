from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from operational.final_refit import scientific_oof_pipeline as pipeline
from operational.final_refit.scientific_fold_heads import fit_scientific_head
from operational.final_refit.scientific_oof_features import physical_phase20_to_state340


def phase_values():
    return {
        "features": np.concatenate(
            (np.full((2, 10), 1, np.float32), np.full((2, 10), 2, np.float32)), 1
        ),
        "rgb_width_logits": np.zeros((2, 2), np.float32),
        "sample_token": np.asarray(["a", "b"]),
        "group_id": np.asarray(["g", "h"]),
        "source_ordinal": np.asarray([4, 7], np.int64),
    }


def test_phase_contract_rgb_then_event_and_target_free_write_once(tmp_path):
    values = phase_values()
    state = physical_phase20_to_state340(values["features"])
    assert np.all(state[:, 160:170] == 1)
    assert np.all(state[:, 330:340] == 2)
    path = tmp_path / "rotations/0/generation/PHASE20/fragment.npz"
    parents = {"R_CTX": "a" * 64, "E_CTX": "b" * 64}
    pipeline.write_fragment(path, values, parents=parents)
    pipeline.write_fragment(path, values, parents=parents)
    values["features"][0, 0] = 3
    with pytest.raises(ValueError, match="changed"):
        pipeline.write_fragment(path, values, parents=parents)


def test_fragment_rejects_target_alias_and_changed_parents(tmp_path):
    path = tmp_path / "rotations/0/generation/PHASE20/fragment.npz"
    values = phase_values()
    with pytest.raises(ValueError, match="target-free"):
        pipeline.write_fragment(path, {**values, "target_ttc_s": np.ones(2)}, parents={})
    pipeline.write_fragment(path, values, parents={"R_CTX": "a" * 64})
    with pytest.raises(ValueError, match="receipt changed"):
        pipeline.write_fragment(path, values, parents={"R_CTX": "b" * 64})


def test_native_target_join_rejects_shifted_or_wrong_group():
    rows = pd.DataFrame({"source_ordinal": [5], "sample_token": ["token5"], "group_id": ["g"]})
    base = SimpleNamespace(
        row=lambda n: {
            "sample_token": "token5",
            "group_id": "g",
            "sequence_id": "s",
            "target_phase": 0.2,
        }
    )
    assert pipeline.target_rows(base, rows)["target_phase"][0] == np.float32(0.2)
    rows.loc[0, "group_id"] = "other"
    with pytest.raises(ValueError, match="crossed"):
        pipeline.target_rows(base, rows)


@pytest.mark.parametrize("updates", [0, -1, 101])
def test_head_rejects_invalid_chunk_before_accessing_any_input(tmp_path, updates):
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="1..100"):
        fit_scientific_head(
            unit_path=Path("absent"),
            recipe=None,
            recipe_path=Path("absent"),
            stage_id="PAIR_E",
            cache=Path("absent"),
            run=tmp_path,
            parent_receipts={},
            admissions={},
            max_updates=updates,
        )
    assert set(tmp_path.iterdir()) == before


def test_guard_preserves_manual_stop_before_monitor_call(tmp_path, monkeypatch):
    class Monitor:
        def __init__(self, *args, **kwargs):
            pass

        def guard(self, *args, **kwargs):
            pytest.fail("manual stop should not access monitor")

    monkeypatch.setattr(pipeline, "SharedResourceMonitor", Monitor)
    (tmp_path / "STOP_REQUEST").write_text("pause")
    with pytest.raises(pipeline.ResourcePause) as caught:
        pipeline.make_guard(tmp_path)()
    assert caught.value.manual is True
    assert (tmp_path / "STOP_REQUEST").read_text() == "pause"
