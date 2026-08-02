from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.run_garl_baseline_suite_v1 import (
    EXPECTED_SEEDS,
    EXPECTED_VARIANTS,
    _git_metadata,
    run_suite,
    validate_suite_config,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_config(tmp_path: Path) -> tuple[Path, dict]:
    release = tmp_path / "release"
    eap = tmp_path / "eap"
    garl = tmp_path / "garl"
    for path in (
        release / "tools/train.py",
        release / "configs/garl_ttc_eventdecoder.yaml",
        release / "configs/ablation/event_lhr.yaml",
        release / "configs/ablation/visual_lhr.yaml",
        eap / "data/train.parquet",
        garl / "data/train.parquet",
        garl / "data/test_inputs.parquet",
        garl / "annotations/train.parquet",
        garl / "splits/train.txt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    (release / ".git").mkdir()
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_names = {
        "event_only": "paper_event_only_lhr.pth",
        "visual_only": "paper_visual_only_lhr.pth",
        "rgbe_late_fusion": "paper_ours_full.pth",
    }
    for variant, name in checkpoint_names.items():
        path = release / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(variant.encode())
        checkpoint_hashes[variant] = _hash(path)

    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "protocol_id": "garlttc_official_v1",
                "seeds": list(EXPECTED_SEEDS),
                "test_labels_available": False,
                "zero_shot_selection_uses_evttc": False,
            }
        ),
        encoding="utf-8",
    )
    readiness = tmp_path / "readiness.json"
    required_gates = {
        "ci_green": True,
        "artifact_contract_green": True,
        "cache_smoke_green": True,
        "cache_gzip_smoke_green": True,
        "signed_metrics_green": True,
        "official_release_audit_green": True,
        "official_preprocessing_parity_green": True,
        "official_model_parity_green": True,
        "cache_pilot_green": True,
        "full_train40_cache_gzip_green": True,
        "long_training_authorized": True,
    }
    _write_json(readiness, {"gates": required_gates})

    release_audit = tmp_path / "release_audit.json"
    _write_json(
        release_audit,
        {
            "artifact_type": "garl_official_release_audit_v1",
            "status": "pass",
            "errors": [],
            "checks": {
                "git": {
                    "commit": "256661242b8a7f5e56aa3c1c02348b30f6e89de6",
                    "status": "pass",
                }
            },
        },
    )
    preprocessing = tmp_path / "preprocessing.json"
    _write_json(
        preprocessing,
        {"artifact_type": "garl_preprocessing_parity_v1", "status": "pass", "samples": 100},
    )
    model = tmp_path / "model.json"
    _write_json(model, {"artifact_type": "garl_model_parity_v1", "status": "pass", "errors": []})

    cache_root = tmp_path / "cache"
    shard = cache_root / "train/shard-00000.pt.gz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"compressed fixture")
    (cache_root / "train/shard-00000.meta.json").write_text("{}", encoding="utf-8")
    cache_manifest = cache_root / "manifest.json"
    protocol_hash = _hash(protocol)
    _write_json(
        cache_manifest,
        {
            "artifact_type": "garlttc_official_lhr_object_cache_v4",
            "schema_version": "garlttc_cache_v4",
            "input_schema": {
                "version": "garlttc_input_v4",
                "event_roi_shape": [2, 20, 128, 128],
            },
            "split_counts": {"train": 71047, "validation": 17697},
            "discard_count": 0,
            "discard_fraction": 0.0,
            "protocol_sha256": protocol_hash,
            "jepa_pair_valid_fraction": 1.0,
            "no_label_fallback": True,
            "uses_official_garl_ttc_labels": True,
            "uses_reconstructed_public_eap_ttc": False,
            "shard_compression": "gzip",
            "shards": [
                {
                    "path": "train/shard-00000.pt.gz",
                    "sha256": _hash(shard),
                }
            ],
        },
    )
    cache_audit = tmp_path / "cache_audit.json"
    _write_json(
        cache_audit,
        {
            "artifact_type": "garlttc_lhr_cache_audit_v2",
            "status": "PASS",
            "errors": [],
            "model_input_fields": ["garl_event_roi", "garl_delta_t_s"],
        },
    )

    config = {
        "suite": {
            "artifact_type": "garl_baseline_suite_v1",
            "phase": 3,
            "track": "official_garl_local",
            "execution_mode": "plan_only",
            "seeds": list(EXPECTED_SEEDS),
            "variants": list(EXPECTED_VARIANTS),
            "selection_metric": "validation_sequence_macro_paper_MiD_overall_signed_v1",
            "test_labels_available": False,
            "test_used_for_selection": False,
        },
        "execution": {
            "execute_training": False,
            "checkpoint_snapshot_epochs": [50],
        },
        "protocol": {"id": "garlttc_official_v1", "path": str(protocol)},
        "paths": {
            "eap_root": str(eap),
            "garlttc_root": str(garl),
            "release_root": str(release),
            "cache_manifest": str(cache_manifest),
            "cache_audit": str(cache_audit),
        },
        "gates": {
            "readiness": str(readiness),
            "release_audit": str(release_audit),
            "preprocessing_parity": str(preprocessing),
            "model_parity": str(model),
        },
        "required_readiness_gates": {"names": list(required_gates)},
        "cache": {
            "expected_train_count": 71047,
            "expected_validation_count": 17697,
            "allowed_compression": ["gzip", "none"],
        },
        "release": {
            "official_commit": "256661242b8a7f5e56aa3c1c02348b30f6e89de6",
            "entrypoint": "tools/train.py",
            "config": "configs/garl_ttc_eventdecoder.yaml",
            "variant_configs": {
                "event_only": "configs/ablation/event_lhr.yaml",
                "visual_only": "configs/ablation/visual_lhr.yaml",
                "rgbe_late_fusion": "configs/garl_ttc_eventdecoder.yaml",
            },
            "checkpoints": {
                variant: {"path": f"checkpoints/{name}", "sha256": checkpoint_hashes[variant]}
                for variant, name in checkpoint_names.items()
            },
        },
        "training": {
            "epochs": 50,
            "batch_size": 128,
            "workers": 2,
            "selection_source": "validation_eap_only",
            "selection_rule": "signed_validation_sequence_macro_mid_then_fr_then_rte",
            "test_selection": False,
        },
        "variant_specs": {
            "event_only": {
                "official_dataset_mode": "event_only",
                "requires_branch_pretraining": True,
            },
            "visual_only": {
                "official_dataset_mode": "image_only",
                "requires_branch_pretraining": True,
            },
            "rgbe_late_fusion": {
                "official_dataset_mode": "image_event",
                "requires_branch_pretraining": True,
            },
        },
    }
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


def test_config_freezes_phase3_contract() -> None:
    config_path = Path("configs/experiment/garl_baseline_suite_v1.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert validate_suite_config(config) == []
    assert tuple(config["suite"]["seeds"]) == EXPECTED_SEEDS
    assert tuple(config["suite"]["variants"]) == EXPECTED_VARIANTS
    assert config["execution"]["execute_training"] is False


def test_invalid_config_rejects_seed_drift_and_execution() -> None:
    config = yaml.safe_load(
        Path("configs/experiment/garl_baseline_suite_v1.yaml").read_text(encoding="utf-8")
    )
    config["suite"]["seeds"] = [7, 13, 21]
    config["execution"]["execute_training"] = True
    errors = validate_suite_config(config)
    assert any("suite.seeds" in error for error in errors)
    assert any("execution.execute_training" in error for error in errors)


def test_git_metadata_hashes_bytes_without_encoding_failure() -> None:
    metadata = _git_metadata(Path(__file__).resolve().parents[2])
    assert metadata["commit"]
    assert metadata["dirty_diff_sha256"]
    assert len(metadata["dirty_diff_sha256"]) == 64


def test_blocked_gates_write_failure_without_metrics_or_training(tmp_path: Path) -> None:
    config_path, config = _valid_config(tmp_path)
    config["required_readiness_gates"]["names"].append("missing_gate")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_suite(config_path, tmp_path / "blocked", repo_root=tmp_path)
    assert result["status"] == "blocked"
    failure = Path(result["failure_path"])
    assert failure.name == "FAILURE.json"
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["training_started"] is False
    assert payload["metrics_available"] is False
    assert not (failure.parent / "baseline_plan.json").exists()


def test_validated_plan_has_three_variants_and_three_seeds_without_execution(
    tmp_path: Path,
) -> None:
    config_path, _ = _valid_config(tmp_path)
    result = run_suite(config_path, tmp_path / "plan", repo_root=tmp_path)
    assert result["status"] == "validated"
    assert result["training_started"] is False
    assert result["metrics_available"] is False
    assert len(result["runs"]) == 9
    assert {(run["variant"], run["seed"]) for run in result["runs"]} == {
        (variant, seed) for variant in EXPECTED_VARIANTS for seed in EXPECTED_SEEDS
    }
    assert all(run["status"] == "planned_not_executed" for run in result["runs"])
    assert not list((tmp_path / "plan").rglob("*.pt"))
    assert not list((tmp_path / "plan").rglob("metrics*.json"))
    assert (tmp_path / "plan" / "baseline_plan.json").is_file()
