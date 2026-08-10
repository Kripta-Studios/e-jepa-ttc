from __future__ import annotations

import numpy as np
import torch

from scripts.analyze_causal_scale_eap_geometry_observability import (
    _activity_observation,
    _relationship,
)


def test_relationship_preserves_distribution_and_linear_slope() -> None:
    reference = np.asarray([1.0, 2.0, 3.0])
    observed = 2.0 * reference + 1.0

    result = _relationship(reference, observed)

    assert result["count"] == 3
    assert np.isclose(result["pearson"], 1.0)
    assert np.isclose(result["slope"], 2.0)
    assert np.isclose(result["reference_std"], np.std(reference))
    assert np.isclose(result["observed_std"], np.std(observed))


def test_activity_observation_recovers_rectangular_moments() -> None:
    events = torch.zeros(1, 1, 2, 8, 10)
    events[:, :, :, 2:6, 3:8] = 1.0

    result = _activity_observation(events)

    assert torch.allclose(result["height"], torch.tensor([[0.5]]), atol=1.0e-6)
    assert torch.allclose(result["width"], torch.tensor([[0.5]]), atol=1.0e-6)
    assert torch.allclose(result["centroid_x"], torch.tensor([[0.55]]), atol=1.0e-6)
    assert torch.allclose(result["centroid_y"], torch.tensor([[0.5]]), atol=1.0e-6)
