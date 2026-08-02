from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
import yaml

import scripts.execute_garl_baseline_suite_v1 as executor


def test_materialize_variant_config_keeps_release_config_unchanged(tmp_path: Path) -> None:
    official = tmp_path / "official.yaml"
    official.write_text(
        yaml.safe_dump(
            {
                "dataset": {"mode": "image_event"},
                "model": {"name": "official"},
                "training": {"batch_size": 2},
                "training_settings": {"snapshot_epochs": [1, 2, 3]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    original = official.read_bytes()
    destination = tmp_path / "run" / "config.yaml"

    digest = executor._materialize_variant_config(
        official,
        destination,
        variant="event_only",
        spec={
            "official_dataset_mode": "event_only",
            "config_overrides": {"training.batch_size": 4},
        },
        snapshot_epochs=[50],
    )

    materialized = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert materialized["dataset"]["mode"] == "event_only"
    assert materialized["training"]["batch_size"] == 4
    assert materialized["training_settings"]["snapshot_epochs"] == [50]
    assert official.read_bytes() == original
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_execute_records_blocked_preflight_without_starting_training(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        executor,
        "run_suite",
        lambda *args, **kwargs: {
            "status": "blocked",
            "training_started": False,
            "metrics_available": False,
            "errors": ["cache gate is red"],
        },
    )

    output = tmp_path / "baseline"
    result = executor.execute(
        suite_config=tmp_path / "suite.yaml",
        output_dir=output,
        eap_root=tmp_path / "eap",
        garlttc_root=tmp_path / "garl_annotations",
        release_root=tmp_path / "release",
        cache_manifest=tmp_path / "manifest.json",
        cache_audit=tmp_path / "audit.json",
        device="cuda",
        max_batches=1,
        variants=("event_only",),
        seeds=(7,),
    )

    assert result["status"] == "blocked_preflight"
    assert result["training_started"] is False
    assert result["metrics_available"] is False
    assert (output / "FAILURE.json").is_file()
    assert not list(output.rglob("stdout.log"))


def test_run_one_passes_absolute_paths_across_release_process_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("run") / "seed-7"
    config_path = run_dir / "config_materialized.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("dataset: {}\n", encoding="utf-8")
    release_root = Path("release")
    release_root.mkdir()
    entrypoint = release_root / "tools" / "train.py"
    eap_root = Path("eap")
    eap_root.mkdir()
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, stdout, stderr, check):
        captured["command"] = command
        captured["cwd"] = cwd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    result = executor._run_one(
        release_root=release_root,
        entrypoint=entrypoint,
        eap_root=eap_root,
        run_dir=run_dir,
        config_path=config_path,
        variant="event_only",
        seed=7,
        epochs=1,
        batch_size=1,
        workers=0,
        device="cpu",
        max_batches=1,
    )

    command = cast(list[str], captured["command"])
    assert isinstance(command, list)
    for option in ("--config", "--data-root", "--output-dir"):
        value = Path(command[command.index(option) + 1])
        assert value.is_absolute()
    assert Path(str(captured["cwd"])).is_absolute()
    assert result["status"] == "completed"


def test_execute_rejects_invalid_smoke_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        executor,
        "run_suite",
        lambda *args, **kwargs: {
            "status": "validated",
            "training_started": False,
            "metrics_available": False,
        },
    )
    with pytest.raises(ValueError, match="batch_size_override"):
        executor.execute(
            suite_config=tmp_path / "suite.yaml",
            output_dir=tmp_path / "baseline",
            eap_root=tmp_path / "eap",
            garlttc_root=tmp_path / "garl_annotations",
            release_root=tmp_path / "release",
            cache_manifest=tmp_path / "manifest.json",
            cache_audit=tmp_path / "audit.json",
            device="cuda",
            max_batches=1,
            batch_size_override=0,
            variants=("event_only",),
            seeds=(7,),
        )


def test_materialize_public_split_is_sequence_disjoint(tmp_path: Path) -> None:
    garl_root = tmp_path / "garl"
    (garl_root / "data").mkdir(parents=True)
    (garl_root / "annotations").mkdir(parents=True)
    rows = pd.DataFrame(
        [
            {"sequence_id": "train-seq", "sample_token": "t1", "ttc": 2.0},
            {"sequence_id": "validation-seq", "sample_token": "v1", "ttc": 4.0},
        ]
    )
    labels = rows[["sequence_id", "sample_token", "ttc"]].copy()
    data_path = garl_root / "data" / "train.parquet"
    labels_path = garl_root / "annotations" / "train.parquet"
    rows.to_parquet(data_path, index=False)
    labels.to_parquet(labels_path, index=False)
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "assignments": {
                    "train": ["train-seq"],
                    "validation": ["validation-seq"],
                }
            }
        ),
        encoding="utf-8",
    )
    cache_manifest = tmp_path / "cache_manifest.json"
    cache_manifest.write_text(json.dumps({"split_path": str(split_path)}), encoding="utf-8")

    report = executor._materialize_public_train_validation_split(
        garlttc_root=garl_root,
        cache_manifest=cache_manifest,
        destination=tmp_path / "protocol_split",
    )

    assert report["train_validation_disjoint"] is True
    assert report["roles"]["train"]["data_rows"] == 1
    assert report["roles"]["validation"]["data_rows"] == 1
    train = pd.read_parquet(report["roles"]["train"]["data_parquet"])
    validation = pd.read_parquet(report["roles"]["validation"]["data_parquet"])
    assert set(train["sequence_id"]) == {"train-seq"}
    assert set(validation["sequence_id"]) == {"validation-seq"}
