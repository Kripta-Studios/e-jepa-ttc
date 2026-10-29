from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from operational.final_refit import event_parity

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_V12 = (
    ROOT.parent / "e-jepa-ttc-v12-efficient-context" / "artifacts" / "train40_system_20261005"
)


def model_contracts() -> tuple[dict[str, object], dict[str, object]]:
    """Return independent minimal contracts for the two supported event arms."""
    a5 = {"hidden_dim": 64, "transport_mode": "legacy", "modality": "event"}
    c2f = {
        "hidden_dim": 64,
        "transport_mode": "adaptive_pyramid",
        "modality": "event",
    }
    protocol = {
        "producers": {
            "a5": {"model_config": copy.deepcopy(a5)},
            "c2f": {"model_config": copy.deepcopy(c2f)},
        }
    }
    port = {
        "producers": {
            "E_A5_MATCHED": {"model_config": copy.deepcopy(a5)},
            "E_C2F_MATCHED": {"model_config": copy.deepcopy(c2f)},
        }
    }
    return protocol, port


def test_event_model_config_comparison_detects_architecture_change() -> None:
    protocol, config = model_contracts()
    assert event_parity.compare_event_model_configs(protocol, config) == {
        "a5": True,
        "c2f": True,
    }
    changed = copy.deepcopy(config)
    changed["producers"]["E_A5_MATCHED"]["model_config"]["hidden_dim"] = 32
    assert event_parity.compare_event_model_configs(protocol, changed) == {
        "a5": False,
        "c2f": True,
    }
    del changed["producers"]["E_C2F_MATCHED"]["model_config"]
    with pytest.raises(ValueError, match="Missing event model config for c2f"):
        event_parity.compare_event_model_configs(protocol, changed)


def test_projection_parity_compares_actual_tensor_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_parity.torch.manual_seed(7)
    model = event_parity.CausalScaleTTC(
        event_parity.CausalScaleTTCConfig(
            hidden_dim=8,
            geometry_dim=16,
            residual_depth=1,
            dropout=0.0,
            transport_enabled=True,
            transport_radius=1,
        )
    )
    model.float().eval().requires_grad_(False)
    events = event_parity.torch.randn((1, 3, 12, 32, 32))
    delta = event_parity.torch.full((1, 2), 0.1)
    assert event_parity._projection_parity(model, events, delta)["all_fields_exact"] is True

    original = event_parity.producer_features

    def changed_projection(*args: object, **kwargs: object) -> dict[str, object]:
        projected = original(*args, **kwargs)
        projected["token128"] = projected["token128"].clone()
        projected["token128"][0, 0] += 1
        return projected

    monkeypatch.setattr(event_parity, "producer_features", changed_projection)
    assert event_parity._projection_parity(model, events, delta)["all_fields_exact"] is False


def test_synthetic_event_reuse_contract_is_scoped_and_label_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise receipt policy without claiming the unavailable historical V12 proof."""
    protocol, config = model_contracts()
    run = tmp_path / "synthetic_v12_contract"
    run.mkdir()
    protocol_path = run / "TRAINING_PROTOCOL.json"
    config_path = tmp_path / "RGB_PORT_PRODUCERS.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    raw_cache = tmp_path / "synthetic_label_isolation_sample.npz"
    np.savez(
        raw_cache,
        events=np.zeros((1, 3, 12, 128, 128), dtype=np.float32),
        delta=np.asarray([0.1], dtype=np.float32),
        forbidden_targets=np.asarray([123.0], dtype=np.float32),
    )

    def load_producer(
        _run: Path, _protocol: dict[str, object], arm: str
    ) -> tuple[object, dict[str, object]]:
        return object(), {
            "checkpoint": f"synthetic://{arm}",
            "checkpoint_sha256": ("a" if arm == "a5" else "c") * 64,
            "committed_updates": event_parity.EXPECTED_UPDATES,
            "state_keys": 1,
            "strict_state_load": True,
        }

    projection = {
        "all_fields_exact": True,
        "field_exact": {"synthetic_projection": True},
        "max_abs_difference": {"synthetic_projection": 0.0},
        "known_shape": [1],
        "support_shape": [1, 2],
    }
    pair_cache = {
        "rows": 3,
        "width": 133,
        "bounded_cpu_replay_rows": 1,
        "bounded_cpu_replay_allclose_atol_1e_5": True,
        "bounded_cpu_replay_max_abs_difference": 0.0,
    }
    monkeypatch.setattr(event_parity, "load_v12_event_producer", load_producer)
    monkeypatch.setattr(event_parity, "_projection_parity", lambda *_args: dict(projection))
    monkeypatch.setattr(event_parity, "_pair_cache_audit", lambda *_args: dict(pair_cache))

    receipt = event_parity.audit_event_reuse(
        v12_run=run,
        rgb_port_config=config_path,
        raw_cache=raw_cache,
        sample_rows=1,
    )
    assert receipt["status"] == "PASSED_WITH_EXPLICIT_ADAPTER"
    assert receipt["inputs"]["targets_loaded"] is False
    assert receipt["inputs"]["optimizer_updates"] == 0
    assert receipt["architecture_config_equal"] == {"a5": True, "c2f": True}
    assert receipt["reuse_decision"]["producer_weights_safe"] is True
    assert receipt["reuse_decision"]["sealed_pair133_cache_safe"] is True
    assert receipt["reuse_decision"]["historical_pair_or_h8_safe"] is False
    assert receipt["reuse_decision"]["known_clock_adapter_required"] is True
    assert receipt["pair_cache"]["rows"] == 3
    assert receipt["pair_cache"]["bounded_cpu_replay_allclose_atol_1e_5"] is True
    assert all(
        endpoint["projection"]["all_fields_exact"] for endpoint in receipt["endpoints"].values()
    )

    rejected = dict(projection)
    rejected["all_fields_exact"] = False
    monkeypatch.setattr(event_parity, "_projection_parity", lambda *_args: dict(rejected))
    with pytest.raises(ValueError, match="projection differs"):
        event_parity.audit_event_reuse(
            v12_run=run,
            rgb_port_config=config_path,
            raw_cache=raw_cache,
            sample_rows=1,
        )

    monkeypatch.setattr(event_parity, "_projection_parity", lambda *_args: dict(projection))
    rejected_cache = dict(pair_cache)
    rejected_cache["bounded_cpu_replay_allclose_atol_1e_5"] = False
    monkeypatch.setattr(event_parity, "_pair_cache_audit", lambda *_args: dict(rejected_cache))
    with pytest.raises(ValueError, match="sealed A5 PAIR feature cache differs"):
        event_parity.audit_event_reuse(
            v12_run=run,
            rgb_port_config=config_path,
            raw_cache=raw_cache,
            sample_rows=1,
        )


@pytest.mark.external_data
def test_external_v12_event_reuse_parity_remains_scoped_and_label_free() -> None:
    """Retain the historical proof when its separately distributed V12 artifacts exist."""
    protocol = EXTERNAL_V12 / "TRAINING_PROTOCOL.json"
    prepare_path = EXTERNAL_V12 / "PREPARE_FREEZE.json"
    if not protocol.is_file() or not prepare_path.is_file():
        pytest.skip("external historical V12 parity artifacts are not installed")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    raw_cache = Path(prepare["cache_root"]) / "shard_00000.npz"
    if not raw_cache.is_file():
        pytest.skip("external V12 bounded raw-cache sample is not installed")
    receipt = event_parity.audit_event_reuse(
        v12_run=EXTERNAL_V12,
        rgb_port_config=ROOT / "configs/rgb_port/producers.json",
        raw_cache=raw_cache,
        sample_rows=1,
    )
    assert receipt["status"] == "PASSED_WITH_EXPLICIT_ADAPTER"
    assert receipt["inputs"]["targets_loaded"] is False
    assert receipt["inputs"]["optimizer_updates"] == 0
    assert receipt["pair_cache"]["rows"] == 88_744
    assert receipt["pair_cache"]["bounded_cpu_replay_allclose_atol_1e_5"] is True
    assert all(
        endpoint["projection"]["all_fields_exact"] for endpoint in receipt["endpoints"].values()
    )
