from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from operational.sota_eval import cmax_reference
from operational.sota_eval.cmax_reference import (
    ReferenceCMaxConfig,
    maximize_reference_cmax,
    signed_iwe,
    warp_affine_expansion_to_reference,
)


def _physical_expansion(
    inverse_ttc: float, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    endpoint = rng.uniform((20.0, 15.0), (76.0, 57.0), size=(80, 2))
    times = np.tile(np.linspace(-0.2, -0.001, 14), len(endpoint))
    endpoints = np.repeat(endpoint, 14, axis=0)
    center = np.asarray((48.0, 36.0))
    # Inverse of x_ref = c + (1-q*dt)*(x_t-c).
    points = center + (endpoints - center) / (1.0 - inverse_ttc * times[:, None])
    centers = np.broadcast_to(center, points.shape).copy()
    polarity = np.repeat(rng.choice((-1, 1), size=len(endpoint)), 14)
    return points, times, polarity, centers


def test_affine_warp_recovers_known_endpoint_coordinates() -> None:
    points, times, _, centers = _physical_expansion(0.5)
    warped = warp_affine_expansion_to_reference(
        points,
        times,
        inverse_ttc_per_s=0.5,
        reference_time_s=0.0,
        reference_center_xy=(48.0, 36.0),
        event_centers_xy=centers,
    )
    endpoint = warped.reshape(80, 14, 2)
    assert np.allclose(endpoint, endpoint[:, :1], atol=1e-12)


def test_reference_cmax_recovers_physical_inverse_ttc() -> None:
    points, times, polarity, centers = _physical_expansion(0.5)
    result = maximize_reference_cmax(
        points,
        times,
        polarity,
        reference_time_s=0.0,
        reference_center_xy=(48.0, 36.0),
        event_centers_xy=centers,
        image_shape=(72, 96),
        config=ReferenceCMaxConfig(minimum_relative_contrast_gain=0.001),
    )
    assert result["valid"] is True
    assert np.isclose(result["inverse_ttc_per_s"], 0.5, atol=0.06)
    assert np.isclose(result["prediction_ttc_s"], 2.0, atol=0.3)


def test_wrong_sign_does_not_align_physical_expansion() -> None:
    points, times, polarity, centers = _physical_expansion(0.5)
    correct = warp_affine_expansion_to_reference(
        points,
        times,
        inverse_ttc_per_s=0.5,
        reference_time_s=0.0,
        reference_center_xy=(48.0, 36.0),
        event_centers_xy=centers,
    )
    wrong_scale = 1.0 + 0.5 * times
    wrong = np.asarray((48.0, 36.0)) + wrong_scale[:, None] * (points - centers)
    correct_iwe, _ = signed_iwe(correct, polarity, image_shape=(72, 96))
    wrong_iwe, _ = signed_iwe(wrong, polarity, image_shape=(72, 96))
    assert np.var(correct_iwe) > np.var(wrong_iwe)


def test_signed_iwe_preserves_signed_mass_for_interior_events() -> None:
    image, survival = signed_iwe(
        np.asarray(((10.25, 11.75), (20.5, 21.5))),
        np.asarray((1, -1)),
        image_shape=(40, 50),
    )
    assert survival == 1.0
    assert np.isclose(image.sum(), 0.0)


def test_runner_resumes_signed_fragments_without_reprediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "queries.json"
    manifest.write_text("{}", encoding="utf-8")
    row = {
        "query_id": "sequence:00",
        "sequence_id": "sequence",
        "anchor_us": 300_000,
        "metadata_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        cmax_reference.baselines,
        "_validate_manifest",
        lambda *_args: ({}, [row]),
    )
    calls = 0

    def predict(*_args: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"valid": True, "prediction_ttc_s": 2.0, "reason": "ok"}

    monkeypatch.setattr(cmax_reference, "_predict_row", predict)
    output = tmp_path / "output"
    first = cmax_reference.run_reference_cmax(
        query_manifest=manifest, output_dir=output
    )
    second = cmax_reference.run_reference_cmax(
        query_manifest=manifest, output_dir=output
    )
    assert calls == 1
    assert first["fragment_sha256"] == second["fragment_sha256"]
    fragment = json.loads(next(output.glob("fragments/*.json")).read_text())
    cmax_reference.baselines._verify_signed(fragment)
