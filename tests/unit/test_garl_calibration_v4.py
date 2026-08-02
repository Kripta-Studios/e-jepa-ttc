from __future__ import annotations

import pytest

from e_jepa_ttc.data.garlttc_calibration import OFFICIAL_FY, CalibrationResolver


def test_official_constant_calibration_is_explicit_and_pair_complete() -> None:
    resolver = CalibrationResolver("official_constant_fy")
    first, second = resolver.resolve_pair({}, (0, 1))
    assert first.fy == pytest.approx(OFFICIAL_FY)
    assert second.fy == pytest.approx(OFFICIAL_FY)
    assert first.calibration_source == "official_constant_fy"
    assert first.join_key is None


def test_unknown_calibration_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported calibration mode"):
        CalibrationResolver("row_fallback")  # type: ignore[arg-type]
