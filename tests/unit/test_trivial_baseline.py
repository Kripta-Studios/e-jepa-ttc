from pathlib import Path

import yaml

from e_jepa_ttc.baselines.trivial import run_trivial_baseline
from e_jepa_ttc.data.evttc import write_manifest
from e_jepa_ttc.data.types import DatasetSequence


def _write_ttc(path: Path, values: list[float]) -> None:
    lines = [f"{idx} {idx:.3f} 1.0 1.0 {value:.3f}" for idx, value in enumerate(values)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_trivial_baseline_uses_train_targets_only(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    _write_ttc(train_dir / "ttc.csv", [2.0, 4.0])
    _write_ttc(test_dir / "ttc.csv", [10.0])

    manifest = tmp_path / "manifest.yaml"
    write_manifest(
        manifest,
        [
            DatasetSequence(
                dataset="EvTTC",
                sequence_id="train-seq",
                local_path=train_dir.as_posix(),
                event_hdf5="events.hdf5",
                ttc_csv="ttc.csv",
                split_group="train-seq",
            ),
            DatasetSequence(
                dataset="EvTTC",
                sequence_id="test-seq",
                local_path=test_dir.as_posix(),
                event_hdf5="events.hdf5",
                ttc_csv="ttc.csv",
                split_group="test-seq",
            ),
        ],
    )
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {"splits": {"train": ["train-seq"], "validation": [], "test": ["test-seq"]}}
        ),
        encoding="utf-8",
    )

    output = run_trivial_baseline(manifest_path=manifest, split_path=split)

    assert output["predictors"]["mean_train_ttc"]["constant_ttc_s"] == 3.0
    assert output["predictors"]["median_train_ttc"]["constant_ttc_s"] == 3.0
    assert output["predictors"]["mean_train_ttc"]["splits"]["test"]["metrics"]["mae_s"] == 7.0
