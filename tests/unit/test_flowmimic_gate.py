from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.utils.io import read_structured
from scripts.evaluate_flowmimic_robustness import (
    _configured_conditions,
    parse_condition,
)
from scripts.run_flowmimic_multiseed import (
    _downstream_command,
    _pretrain_command,
)
from scripts.summarize_flowmimic_multiseed import _seed_bootstrap

CONFIG_PATH = Path("configs/experiment/flowmimic_e0_e1_multiseed.yaml")


def test_frozen_gate_commands_keep_test_closed_and_pair_only_alignment() -> None:
    config = read_structured(CONFIG_PATH)
    e0 = _pretrain_command(
        config,
        variant="E0",
        seed=13,
        output_dir=Path("artifacts/runs/e0"),
    )
    e1 = _pretrain_command(
        config,
        variant="E1",
        seed=13,
        output_dir=Path("artifacts/runs/e1"),
    )
    downstream = _downstream_command(
        config,
        seed=13,
        pretrained_checkpoint=Path("artifacts/runs/e1/jepa_encoder_best.pt"),
        output_dir=Path("artifacts/runs/e1_ft"),
    )

    e0_alignment = e0[e0.index("--flowmimic-alignment-weight") + 1]
    e1_alignment = e1[e1.index("--flowmimic-alignment-weight") + 1]
    assert e0_alignment == "0.0"
    assert e1_alignment == "0.25"
    assert e0[e0.index("--epochs") + 1] == "30"
    assert e1[e1.index("--epochs") + 1] == "30"
    assert downstream[downstream.index("--epochs") + 1] == "30"
    assert "test" not in downstream
    assert "--allow-final-test-evaluation" not in downstream


def test_full_robustness_matrix_has_22_nonclean_conditions() -> None:
    config = read_structured(CONFIG_PATH)
    conditions = _configured_conditions(config, None)

    assert len(conditions) == 22
    assert ("event_dropout", 0.7) in conditions
    assert ("temporal_window_scale", 1.5) in conditions
    assert ("polarity_drop_positive", 1.0) in conditions


def test_condition_parser_validates_kind_and_severity() -> None:
    assert parse_condition("timestamp_jitter_us=200") == (
        "timestamp_jitter_us",
        200.0,
    )
    with pytest.raises(argparse.ArgumentTypeError):
        parse_condition("none=0")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_condition("event_dropout=2")


def test_paired_seed_bootstrap_is_reproducible() -> None:
    values = np.asarray([-0.1, -0.2, -0.3])
    first = _seed_bootstrap(values, iterations=500)
    second = _seed_bootstrap(values, iterations=500)

    assert first == second
    assert first["estimate"] == pytest.approx(-0.2)
    assert first["upper"] < 0.0
