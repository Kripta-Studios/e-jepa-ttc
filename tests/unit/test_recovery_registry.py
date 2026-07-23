from __future__ import annotations

import pytest

from scripts.register_recovery_run import _selection_criterion


def test_recovery_selection_criteria_are_stage_exact() -> None:
    assert _selection_criterion("ssl_pretrain") == "validation_loss"
    assert _selection_criterion("downstream_ttc") == "validation_mae_s"
    with pytest.raises(ValueError, match="Unsupported recovery stage"):
        _selection_criterion("unknown")
