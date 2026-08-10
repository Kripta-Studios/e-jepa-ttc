from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "a5_freeze_v2", ROOT / "scripts" / "freeze_a5_suite_configs.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v2_selection_rewrites_runtime_model_and_contract(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    model_path = repo / "configs" / "model" / "base.yaml"
    model_path.parent.mkdir(parents=True)
    model_path.write_text(
        yaml.safe_dump(
            {
                "transport_enabled": True,
                "transport_radius": 4,
                "transport_temperature": 0.07,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    preflight = repo / "artifacts" / "metrics" / "v2.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(
        json.dumps(
            {
                "artifact_type": "a5_transport_preflight_train_only_v2",
                "artifact_sha256": "abc",
                "scope": {
                    "public_train_only": True,
                    "validation_or_test_opened": False,
                    "optimizer_steps": 0,
                },
                "decision": {
                    "a5_corr_authorized": True,
                    "selected_radius": 2,
                    "selected_temperature": 0.04,
                },
            }
        ),
        encoding="utf-8",
    )
    original_root = MODULE.ROOT
    MODULE.ROOT = repo
    try:
        selection = MODULE._load_transport_selection(preflight)
        assert selection is not None
        payload = {
            "model_config": "configs/model/base.yaml",
            "decision_contract": {
                "representation_change": {
                    "transport_radius": 4,
                    "transport_candidates_per_position": 81,
                }
            },
        }
        output_dir = repo / "artifacts" / "configs" / "runtime"
        output_dir.mkdir(parents=True)
        runtime = MODULE._apply_transport_selection(
            payload,
            output_dir=output_dir,
            name="seed7",
            selection=selection,
        )
        assert runtime == "artifacts/configs/runtime/model_seed7.yaml"
        frozen_model = yaml.safe_load((output_dir / "model_seed7.yaml").read_text())
        assert frozen_model["transport_radius"] == 2
        assert frozen_model["transport_temperature"] == 0.04
        change = payload["decision_contract"]["representation_change"]
        assert change["transport_candidates_per_position"] == 25
        assert payload["decision_contract"]["preflight_contract"]["selected_radius"] == 2
    finally:
        MODULE.ROOT = original_root


def test_v3_confirmation_selection_is_accepted(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    preflight = repo / "artifacts" / "metrics" / "v3.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(
        json.dumps(
            {
                "artifact_type": "a5_transport_preflight_train_only_v3_confirmation",
                "artifact_sha256": "def",
                "scope": {
                    "public_train_only": True,
                    "validation_or_test_opened": False,
                    "optimizer_steps": 0,
                },
                "decision": {
                    "a5_corr_authorized": True,
                    "selected_radius": 1,
                    "selected_temperature": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    original_root = MODULE.ROOT
    MODULE.ROOT = repo
    try:
        selection = MODULE._load_transport_selection(preflight)
        assert selection is not None
        assert selection["artifact_type"] == "a5_transport_preflight_train_only_v3_confirmation"
        assert selection["radius"] == 1
        assert selection["temperature"] == 0.02
    finally:
        MODULE.ROOT = original_root
