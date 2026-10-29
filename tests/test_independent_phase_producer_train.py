from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any

import pytest

from operational.independent_refit import producer_train as subject
from operational.independent_refit.producer_train import epoch_order

torch: Any = importlib.import_module("torch")


def test_epoch_order_keeps_every_history_valid_query() -> None:
    groups = (tuple(range(0, 17_524)), tuple(range(17_524, 35_048)))
    generator = torch.Generator(device="cpu").manual_seed(7)
    order = epoch_order(groups, generator, population_size=35_048)
    assert len(order) == 35_048
    assert sorted(order) == list(range(35_048))
    assert len(order[-8:]) == 8
    assert (len(order) + 31) // 32 == 1_096


@pytest.mark.parametrize("crash_on_checkpoint", [1, 2])
def test_curve_transaction_recovers_both_crash_boundaries(
    tmp_path: Any, crash_on_checkpoint: int
) -> None:
    cursor: dict[str, Any] = {
        "curve": [{"update": 100, "losses": {"total": 1.0}}],
        "pending_curve": [],
        "pending_curve_update": 0,
    }
    durable: dict[str, Any] = deepcopy(cursor)
    calls = 0

    def checkpoint() -> None:
        nonlocal calls, durable
        calls += 1
        if calls == crash_on_checkpoint:
            if crash_on_checkpoint == 1:
                durable = deepcopy(cursor)
            raise RuntimeError("simulated crash")
        durable = deepcopy(cursor)

    with pytest.raises(RuntimeError, match="simulated crash"):
        subject._durable_curve_transaction(tmp_path, "contract", cursor, checkpoint)

    resumed = deepcopy(durable)

    def resumed_checkpoint() -> None:
        nonlocal durable
        durable = deepcopy(resumed)

    subject._durable_curve_transaction(
        tmp_path, "contract", resumed, resumed_checkpoint
    )
    assert durable["curve"] == []
    assert durable["pending_curve"] == []
    artifact = tmp_path / "curves/curve_000100.json"
    assert artifact.exists()
    assert subject._json(artifact)["first_update"] == 100
