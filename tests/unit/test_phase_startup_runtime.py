import numpy as np
import pytest
import torch

from operational.phase_startup.model import ExpertRouter
from operational.phase_startup.runtime import (
    RouterEnsemble,
    describe_prediction,
    load_bundle_router,
    load_router,
    predict_history,
)


def test_ensemble_fallback_is_bitwise_and_history_is_causal():
    model = RouterEnsemble([ExpertRouter(coordinate="inverse") for _ in range(3)])
    native = torch.tensor([[1.234567, 2.0, 3.0], [60.00001, 0.0, 0.0]])
    absent = torch.full((2,), float("nan"))
    q = torch.zeros(2, 20)
    assert torch.equal(model(native, absent, q), native[:, 0])
    script = torch.jit.script(model)
    assert torch.equal(script(native, absent, q), native[:, 0])
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 11.0, 11.0], [0.0, 0.0, 999.0, 999.0]])
    clocks = np.array([0, 100000, 200000])
    first, _ = predict_history(model, native[0].numpy(), boxes, clocks, 100000)
    other, _ = predict_history(model, native[0].numpy(), boxes[:2], clocks[:2], 100000)
    assert first == other


def test_loader_rejects_unbound_configuration(tmp_path):
    import json

    (tmp_path / "FINAL_FREEZE.json").write_text(json.dumps({"config_sha256": "bad"}))
    (tmp_path / "CONFIG.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration"):
        load_router(tmp_path)


def test_stale_history_cannot_reroute_native_experts():
    model = RouterEnsemble([ExpertRouter(coordinate="inverse") for _ in range(3)])
    native = torch.tensor([[1.234567, 20.0, 30.0]])
    quality = torch.ones(1, 20)
    quality[:, 11] = 0.2
    assert torch.equal(model(native, torch.tensor([float("nan")]), quality), native[:, 0])


def test_deployed_interval_matches_calibration_clipped_geometry_convention():
    native = np.array([1.0, 100.0, 200.0], np.float32)
    physical = {"valid": True, "ttc": 1000.0, "count": 8, "reason": "available"}
    result = describe_prediction(
        10.0, native, physical, {"constant_radius": 0.0, "scaled_factor": 1.0}
    )
    assert result["interval_80_seconds"] == [-39.5, 59.5]


def test_bundle_rejects_a_different_variant_freeze(tmp_path):
    import json
    from pathlib import Path

    from operational.evttc_transfer_audit.common import file_sha

    (tmp_path / "upstream").mkdir()
    (tmp_path / "upstream/BUNDLE.json").write_text("{}")
    (tmp_path / "evaluated").mkdir()
    freeze = tmp_path / "evaluated/FINAL_FREEZE.json"
    freeze.write_text('{"variant":"evaluated"}')
    binding = {
        "schema": "phase_startup_bundle_v1",
        "implementation": {
            name: file_sha(Path("operational/phase_startup") / name)
            for name in ("model.py", "geometry.py", "runtime.py")
        },
        "upstream_sha256": file_sha(tmp_path / "upstream/BUNDLE.json"),
        "freezes": {"evaluated": file_sha(freeze)},
    }
    (tmp_path / "ROUTER_BINDING.json").write_text(json.dumps(binding))
    freeze.write_text('{"variant":"all_data_refit"}')
    with pytest.raises(ValueError, match="variant identity"):
        load_bundle_router(tmp_path, "evaluated")


def test_fit_is_independent_of_excluded_labels_and_inputs(tmp_path):
    from scripts.train_phase_startup import fit_router

    a = {
        "native": np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [30.0, 40.0, 50.0]], np.float32),
        "geometry": np.array([1.0, 2.0, 3.0], np.float32),
        "quality": np.ones((3, 20), np.float32),
        "target": np.array([1.1, 2.2, 999.0], np.float32),
        "family": np.array(["A", "B", "heldout"]),
        "group": np.array(["a", "b", "heldout"]),
    }
    mask = np.array([True, True, False])
    torch.set_num_threads(2)
    first = fit_router(a, mask, "QA", 7, tmp_path / "first", "inverse")
    for key in ("native", "geometry", "quality", "target"):
        a[key][-1] = -12345
    second = fit_router(a, mask, "QA", 7, tmp_path / "second", "inverse")
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])
