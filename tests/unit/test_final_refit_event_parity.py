from __future__ import annotations

import copy
import json
from pathlib import Path

from operational.final_refit.event_parity import (
    audit_event_reuse,
    compare_event_model_configs,
)

ROOT = Path(__file__).resolve().parents[2]
V12 = ROOT.parent / "e-jepa-ttc-v12-efficient-context" / "artifacts" / "train40_system_20261005"


def test_event_model_config_comparison_detects_architecture_change() -> None:
    protocol = json.loads((V12 / "TRAINING_PROTOCOL.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/rgb_port/producers.json").read_text(encoding="utf-8"))
    assert compare_event_model_configs(protocol, config) == {"a5": True, "c2f": True}
    changed = copy.deepcopy(config)
    changed["producers"]["E_A5_MATCHED"]["model_config"]["hidden_dim"] = 32
    assert compare_event_model_configs(protocol, changed) == {"a5": False, "c2f": True}


def test_real_v12_event_reuse_parity_is_scoped_and_label_free() -> None:
    prepare = json.loads((V12 / "PREPARE_FREEZE.json").read_text(encoding="utf-8"))
    receipt = audit_event_reuse(
        v12_run=V12,
        rgb_port_config=ROOT / "configs/rgb_port/producers.json",
        raw_cache=Path(prepare["cache_root"]) / "shard_00000.npz",
        sample_rows=1,
    )
    assert receipt["status"] == "PASSED_WITH_EXPLICIT_ADAPTER"
    assert receipt["inputs"]["targets_loaded"] is False
    assert receipt["inputs"]["optimizer_updates"] == 0
    assert receipt["reuse_decision"]["producer_weights_safe"] is True
    assert receipt["reuse_decision"]["sealed_pair133_cache_safe"] is True
    assert receipt["reuse_decision"]["historical_pair_or_h8_safe"] is False
    assert receipt["pair_cache"]["rows"] == 88_744
    assert receipt["pair_cache"]["bounded_cpu_replay_allclose_atol_1e_5"] is True
    assert all(
        endpoint["projection"]["all_fields_exact"] for endpoint in receipt["endpoints"].values()
    )
