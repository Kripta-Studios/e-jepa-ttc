from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.run_garl_matched_screen import materialize_config

ROOT = Path(__file__).resolve().parents[2]


def _subset(tmp_path: Path, train_rows: int, validation_rows: int = 2048) -> Path:
    payload = {
        "artifact_type": "garl_event_only_matched_screen_subset_v1",
        "artifact_sha256": "unit-test",
        "roles": {
            "train": {
                "rows": train_rows,
                "assets": {"path": "train_assets.txt"},
                "data": {"path": "train_data.parquet"},
                "labels": {"path": "train_labels.parquet"},
            },
            "validation": {
                "rows": validation_rows,
                "assets": {"path": "validation_assets.txt"},
                "data": {"path": "validation_data.parquet"},
                "labels": {"path": "validation_labels.parquet"},
            },
        },
    }
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _official_config(tmp_path: Path) -> Path:
    path = tmp_path / "official.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {},
                "model": {},
                "training_settings": {},
                "dirs": {},
                "cudnn": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_garl_materialized_config_accepts_preregistered_8192_budget(tmp_path: Path) -> None:
    out = tmp_path / "out.yaml"
    config = materialize_config(
        official_config=_official_config(tmp_path),
        subset_manifest=_subset(tmp_path, 8192),
        release_root=tmp_path / "release",
        eap_root=tmp_path / "eap",
        output_config=out,
        output_dir=tmp_path / "run",
        epochs=18,
        batch_size=32,
        num_workers=0,
        minimum_selection_epoch=8,
    )
    assert config["dataset"]["db_sample_size"] == 8192
    assert config["matched_protocol"]["train_rows"] == 8192
    assert config["matched_protocol"]["validation_rows"] == 2048


def test_hardware_rescue_preserves_effective_batch_and_scientific_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source_payload = {
        "experiment": {"name": "x"},
        "model_config": "configs/model/example.yaml",
        "data": {"manifest": "immutable"},
        "training": {
            "seed": 7,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "prefetch_factor": 2,
        },
        "loss": {"weight": 8.0},
        "decision_contract": {"frozen_gate": True},
    }
    source.write_text(yaml.safe_dump(source_payload, sort_keys=False), encoding="utf-8")
    out = tmp_path / "rescue.yaml"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/freeze_hardware_rescue_config.py",
            "--source-config",
            str(source),
            "--output-config",
            str(out),
            "--seed",
            "13",
            "--batch-size",
            "16",
            "--gradient-accumulation-steps",
            "2",
            "--num-workers",
            "0",
            "--prefetch-factor",
            "2",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    rescue = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert rescue["training"]["batch_size"] == 16
    assert rescue["training"]["gradient_accumulation_steps"] == 2
    assert rescue["training"]["seed"] == 13
    assert rescue["data"] == source_payload["data"]
    assert rescue["loss"] == source_payload["loss"]
    assert rescue["decision_contract"]["frozen_gate"] is True
    assert rescue["decision_contract"]["hardware_rescue_selection_source"] == (
        "CUDA_feasibility_only_not_validation"
    )
