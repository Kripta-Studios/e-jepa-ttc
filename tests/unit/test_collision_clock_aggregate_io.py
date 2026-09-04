from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.collision_clock_aggregate import _read_oof_csv
from e_jepa_ttc.evaluation.collision_clock_cross_arm import (
    _read_oof_csv as _read_cross_arm_oof_csv,
)


def test_oof_csv_round_trip_parser_preserves_canonical_float64(tmp_path: Path) -> None:
    """Protect the signed target identity across the runner/aggregate boundary."""

    original = np.float64(3.9558374881744394)
    path = tmp_path / "oof.csv"
    pd.DataFrame({"target_ttc_s": [original]}).to_csv(path, index=False)
    default_value = pd.read_csv(path)["target_ttc_s"].iloc[0]
    restored = _read_oof_csv(path)["target_ttc_s"].iloc[0]
    cross_arm_restored = _read_cross_arm_oof_csv(path)["target_ttc_s"].iloc[0]
    assert default_value != original
    assert restored == original
    assert cross_arm_restored == original
