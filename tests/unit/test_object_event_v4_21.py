from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.training.object_event_v4_21 import (
    ObjectEventV421AuditConfig,
    box_scale_proxies,
    pearson,
    train_orientation,
)


def _boxes() -> torch.Tensor:
    return torch.tensor(
        [
            [[10.0, 10.0, 30.0, 30.0], [10.0, 10.0, 30.0, 30.0], [8.0, 8.0, 32.0, 32.0]],
            [[10.0, 10.0, 30.0, 30.0], [10.0, 10.0, 30.0, 30.0], [12.0, 12.0, 28.0, 28.0]],
        ]
    )


def test_box_scale_proxies_detect_expansion_and_contraction() -> None:
    values = box_scale_proxies(_boxes())
    assert values["box_geometric_log_scale"][0] > 0.0
    assert values["box_geometric_log_scale"][1] < 0.0


def test_isotropic_resize_has_zero_anisotropy() -> None:
    values = box_scale_proxies(_boxes())
    assert np.allclose(values["box_log_scale_anisotropy"], 0.0, atol=1e-7)


def test_centered_resize_has_zero_translation() -> None:
    values = box_scale_proxies(_boxes())
    assert np.allclose(values["box_normalized_translation"], 0.0, atol=1e-7)


def test_train_orientation_flips_negative_correlation() -> None:
    target = np.array([-1.0, 0.0, 1.0])
    score = -target
    assert train_orientation(score, target) == -1.0
    assert pearson(-score, target) > 0.99


def test_pearson_handles_constant_input() -> None:
    assert pearson(np.ones(4), np.arange(4.0)) == 0.0


def test_config_validation() -> None:
    try:
        ObjectEventV421AuditConfig(map_size=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
