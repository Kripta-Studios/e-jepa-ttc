from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import scripts.run_garl_matched_screen as runner


def _fixture(tmp_path: Path, *, train_rows: int = 2048) -> tuple[Path, Path]:
    official = tmp_path / "event_lhr.yaml"
    official.write_text(
        yaml.safe_dump(
            {
                "exp_type": "event_lhr",
                "dirs": {"output": "release-output"},
                "cudnn": {"enabled": True, "deterministic": False, "benchmark": False},
                "dataset": {"root": "release-data", "mode": "event_only"},
                "model": {
                    "pretrained_ckpt_rgb": "paper-rgb.pth",
                    "pretrained_ckpt_event": "paper-event.pth",
                },
                "training_settings": {"total_epochs": 50, "batch_size": 128},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    roles = {}
    for role, rows, sequences in (
        ("train", train_rows, ["train-seq"]),
        ("validation", 2048, ["validation-seq"]),
    ):
        for suffix in ("data.parquet", "labels.parquet", "assets.txt"):
            (tmp_path / f"{role}_{suffix}").write_bytes(b"fixture")
        roles[role] = {
            "rows": rows,
            "sequences": sequences,
            "assets": {"path": f"{role}_assets.txt"},
            "data": {"path": f"{role}_data.parquet"},
            "labels": {"path": f"{role}_labels.parquet"},
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "garl_event_only_matched_screen_subset_v1",
                "artifact_sha256": "a" * 64,
                "roles": roles,
            }
        ),
        encoding="utf-8",
    )
    return official, manifest


def test_materialize_config_disables_exposed_pretraining_and_binds_exact_roles(
    tmp_path: Path,
) -> None:
    official, manifest = _fixture(tmp_path)
    output = tmp_path / "run" / "matched.yaml"

    payload = runner.materialize_config(
        official_config=official,
        subset_manifest=manifest,
        release_root=tmp_path / "release",
        eap_root=tmp_path / "eap",
        output_config=output,
        output_dir=tmp_path / "output",
        epochs=18,
        batch_size=32,
        num_workers=8,
        minimum_selection_epoch=8,
    )

    assert payload["model"]["pretrained_ckpt_event"] == ""
    assert payload["model"]["pretrained_ckpt_rgb"] == ""
    assert payload["matched_protocol"]["from_scratch"] is True
    assert payload["matched_protocol"]["release_checkpoint_initialization"] is False
    assert payload["dataset"]["train"]["data_parquet"].endswith("train_data.parquet")
    assert payload["dataset"]["test"]["data_parquet"].endswith(
        "validation_data.parquet"
    )
    assert payload["training_settings"]["snapshot_epochs"] == list(range(8, 19))
    assert output.is_file()
    assert official.read_text(encoding="utf-8").find("paper-event.pth") >= 0


def test_materialize_config_rejects_nonmatched_row_count(tmp_path: Path) -> None:
    official, manifest = _fixture(tmp_path, train_rows=2047)

    with pytest.raises(ValueError, match="exact 2048/2048"):
        runner.materialize_config(
            official_config=official,
            subset_manifest=manifest,
            release_root=tmp_path / "release",
            eap_root=tmp_path / "eap",
            output_config=tmp_path / "matched.yaml",
            output_dir=tmp_path / "output",
            epochs=18,
            batch_size=32,
            num_workers=8,
            minimum_selection_epoch=8,
        )
