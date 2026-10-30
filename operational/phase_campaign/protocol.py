"""Recording exclusions and conservative scenario-family holdouts."""

from __future__ import annotations

from typing import Any

import numpy as np


def family_folds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hold out whole families without claiming acquisition independence."""
    names = [r["sequence_id"] for r in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate recording identity")
    admitted = [r for r in records if r["role"] == "development_exposed"]
    families = sorted({r["scenario_family"] for r in admitted})
    if len(families) < 2:
        raise ValueError("At least two eligible families are required")
    return [
        {
            "family": family,
            "train": sorted(r["sequence_id"] for r in admitted if r["scenario_family"] != family),
            "test": sorted(r["sequence_id"] for r in admitted if r["scenario_family"] == family),
        }
        for family in families
    ]


def fold_indices(values: dict[str, np.ndarray], fold: dict) -> tuple[np.ndarray, np.ndarray]:
    """Select source TRAIN40 and eligible development groups without fold leakage."""
    if set(fold["train"]) & set(fold["test"]):
        raise ValueError("Train/test recording overlap")
    domain, group = values["domain"], values["group_id"]
    train = np.flatnonzero(
        (domain == "eap") | ((domain == "evttc") & np.isin(group, fold["train"]))
    )
    test = np.flatnonzero((domain == "evttc") & np.isin(group, fold["test"]))
    if not len(train) or not len(test) or np.intersect1d(train, test).size:
        raise ValueError("Empty fold or train/test index overlap")
    return train, test
