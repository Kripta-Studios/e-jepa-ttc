"""Zero-optimizer tests for frozen RGB-PORT head prediction behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.rgb_port.evaluate import HEAD_IDS
from operational.rgb_port.predict_heads import _forward, _load_inputs, _prediction_source


def _temporal_cache(expert_phase: np.ndarray) -> dict[str, np.ndarray]:
    count = len(expert_phase)
    return {
        "features": np.zeros((count, 1, 17), np.float32),
        "timing": np.zeros((count, 1, 4), np.float32),
        "valid": np.ones((count, 1), np.bool_),
        "expert_phase": expert_phase.astype(np.float32),
        "sample_token": np.asarray([f"s{i}" for i in range(count)]),
    }


def test_actual_temporal_head_retains_negative_prediction_sign() -> None:
    model = TemporalRefiner(TemporalConfig(feature_count=17, hidden=160)).eval()
    cache = _temporal_cache(np.asarray([[-0.3, -0.2, -0.1]], np.float32))
    result = _forward(model, "E_CTX_MATCHED", cache)
    assert result["prediction_ttc"].shape == (1,)
    assert result["prediction_ttc"][0] < 0
    assert np.isfinite(result["q10_ttc"]).all() and np.isfinite(result["q90_ttc"]).all()


def test_actual_fusion_head_outputs_all_native_phase_coordinates() -> None:
    model = DualClockFusion("F_TRUE").eval()
    cache = {
        "event_features": np.zeros((2, 1, 17), np.float32),
        "event_timing": np.zeros((2, 1, 4), np.float32),
        "event_valid": np.ones((2, 1), np.bool_),
        "event_expert_phase": np.asarray([[0.1, 0.2, 0.3], [-0.3, -0.2, -0.1]], np.float32),
        "rgb_features": np.zeros((2, 1, 17), np.float32),
        "rgb_timing": np.zeros((2, 1, 4), np.float32),
        "rgb_valid": np.ones((2, 1), np.bool_),
        "sample_token": np.asarray(["a", "b"]),
    }
    result = _forward(model, "F_TRUE", cache)
    assert set(result) == {
        "point_phase",
        "raw_location",
        "q10",
        "q90",
        "prediction_ttc",
        "q10_ttc",
        "q90_ttc",
    }
    assert result["prediction_ttc"][0] > 0 and result["prediction_ttc"][1] < 0


def test_missing_rgb_selects_exact_event_context_source() -> None:
    indexed = {
        fit_id: ({"available": 0} if fit_id != "E_CTX_MATCHED" else {"available": 0, "missing": 1})
        for fit_id in HEAD_IDS
    }
    assert _prediction_source("F_TRUE", "missing", indexed) == ("E_CTX_MATCHED", 1, True)
    assert _prediction_source("R_CTX", "missing", indexed) == ("R_CTX", None, False)


def test_inputs_json_requires_all_six_fits_and_preserves_paths(tmp_path: Path) -> None:
    paths = {}
    for fit_id in HEAD_IDS:
        path = tmp_path / f"{fit_id}.npz"
        path.write_bytes(b"cache")
        paths[fit_id] = str(path)
    source = tmp_path / "inputs.json"
    source.write_text(json.dumps(paths), encoding="utf-8")
    loaded = _load_inputs(source)
    assert set(loaded) == HEAD_IDS
    paths.pop("F_ZERO")
    source.write_text(json.dumps(paths), encoding="utf-8")
    try:
        _load_inputs(source)
    except ValueError as error:
        assert "exactly" in str(error)
    else:
        raise AssertionError("incomplete six-head freeze was accepted")
