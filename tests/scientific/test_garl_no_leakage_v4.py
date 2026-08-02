"""Scientific leakage guards for the predict/score and split contracts."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from e_jepa_ttc.data.evttc_garl_adapter import (
    make_garl_model_input,
    reject_labels_from_predict_payload,
)
from e_jepa_ttc.data.split import validate_split_groups
from e_jepa_ttc.data.types import DatasetSequence
from e_jepa_ttc.evaluation.garl_ttc_protocol import select_checkpoint
from e_jepa_ttc.training import eap_jepa


def _model_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(2, 2, 20, 128, 128),
        torch.tensor([[100, 200], [300, 400]], dtype=torch.int64),
        torch.tensor([0.1, 0.1]),
    )


def _ssl_media() -> pd.DataFrame:
    timestamps = [0, 100_000, 200_000, 300_000, 400_000, 500_000]
    return pd.DataFrame(
        {
            "sequence_id": ["sequence-a"] * len(timestamps),
            "rgb_exposure_start_timestamp_us": timestamps,
            "rgb_exposure_end_timestamp_us": timestamps,
        }
    )


def _ssl_config() -> eap_jepa.EAPJEPATrainerConfig:
    return eap_jepa.EAPJEPATrainerConfig(
        horizons_ms=(100,),
        max_windows_per_sequence=3,
        geometry_loss_weight=0.0,
    )


def test_predict_payload_rejects_future_and_ttc_labels() -> None:
    with pytest.raises(ValueError, match="forbidden labels"):
        reject_labels_from_predict_payload({"events": object(), "frame_ttc": 1.0})


def test_model_input_constructor_has_no_label_argument_and_validates_shapes() -> None:
    events, timestamps, delta = _model_input()
    model_input = make_garl_model_input(events, timestamps, delta)
    model_input.validate()
    with pytest.raises(ValueError, match="event_roi_endpoints"):
        make_garl_model_input(events[:, :, :19], timestamps, delta)


def test_sequence_split_groups_cannot_overlap() -> None:
    sequences = [
        DatasetSequence("set", "train-seq", ".", "events.h5", split_group="group-a"),
        DatasetSequence("set", "val-seq", ".", "events.h5", split_group="group-b"),
    ]
    with pytest.raises(ValueError, match="appears in train and validation"):
        validate_split_groups(
            sequences,
            {"train": ["train-seq"], "validation": ["val-seq", "train-seq"]},
        )
    overlapping_groups = [
        sequences[0],
        DatasetSequence("set", "val-seq", ".", "events.h5", split_group="group-a"),
    ]
    with pytest.raises(ValueError, match="Split group"):
        validate_split_groups(
            overlapping_groups,
            {"train": ["train-seq"], "validation": ["val-seq"]},
        )


def test_checkpoint_selection_requires_the_signed_validation_protocol() -> None:
    metrics = [
        {
            "protocol": "garl_signed_v1",
            "paper_MiD_overall": 2.0,
            "failure_rate_pct": 1.0,
        },
        {
            "protocol": "garl_signed_v1",
            "paper_MiD_overall": 1.0,
            "failure_rate_pct": 2.0,
        },
    ]
    selected = select_checkpoint(metrics)
    assert selected["selected_index"] == 1
    with pytest.raises(ValueError, match="different protocol"):
        select_checkpoint([{"protocol": "evttc", "paper_MiD_overall": 0.0}])


def test_ssl_pure_sampler_ids_do_not_change_when_labels_mutate(monkeypatch) -> None:
    monkeypatch.setattr(eap_jepa, "load_eap_media_table", lambda *_args, **_kwargs: _ssl_media())
    label_versions = [
        pd.DataFrame({"frame_ttc": [0.1, 9.0], "category": ["car", "person"]}),
        pd.DataFrame({"frame_ttc": [-50.0, 500.0], "category": ["mutated", "labels"]}),
    ]
    observed_ids: list[list[tuple[str, int]]] = []
    for labels in label_versions:
        monkeypatch.setattr(
            eap_jepa,
            "load_eap_sequence_labels",
            lambda *_args, labels=labels, **_kwargs: labels,
        )
        dataset = eap_jepa.EAPOnDemandJEPADataset("unused", ["sequence-a"], _ssl_config())
        observed_ids.append(
            [(sample.sequence_id, sample.timestamp_us) for sample in dataset.samples]
        )
        dataset.close()

    assert observed_ids[0] == observed_ids[1]


def test_ssl_pure_sampler_never_calls_label_loader(monkeypatch) -> None:
    monkeypatch.setattr(eap_jepa, "load_eap_media_table", lambda *_args, **_kwargs: _ssl_media())

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("SSL-Pure must not call load_eap_sequence_labels")

    monkeypatch.setattr(eap_jepa, "load_eap_sequence_labels", fail_if_called)
    dataset = eap_jepa.EAPOnDemandJEPADataset("unused", ["sequence-a"], _ssl_config())
    assert dataset.samples
    dataset.close()
