"""Exactness tests for the additive TRAIN40 constant-check fast path."""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4 as original_collate
from operational.train40_system.fast_collate import (
    _all_event_rows_constant,
    collate_object_event_v4,
)


def _record(token: str, *, steps: int = 3) -> dict[str, Any]:
    events = np.arange(steps * 12 * 4 * 4, dtype=np.float32).reshape(steps, 12, 4, 4)
    return {
        "event_v4_common_roi": events,
        "_event_v4_expected_steps": steps,
        "garl_delta_t_s": np.float32(0.1),
        "observable_motion": np.arange(18, dtype=np.float32),
        "garl_visible_heights_px": np.asarray([3.0, 4.0], dtype=np.float32),
        "ttc_s": np.float32(-2.0),
        "event_v4_boxes_xyxy": np.tile(
            np.asarray([0.0, 0.0, 3.0, 3.0], dtype=np.float32), (steps, 1)
        ),
        "event_v4_common_square_xyxy": np.asarray(
            [0.0, 0.0, 4.0, 4.0], dtype=np.float32
        ),
        "sequence_id": "sequence",
        "sample_token": token,
        "track_id": "track",
    }


def _assert_same_batch(left: Any, right: Any) -> None:
    for field in fields(left):
        first = getattr(left, field.name)
        second = getattr(right, field.name)
        if isinstance(first, torch.Tensor):
            assert torch.equal(first, second), field.name
        else:
            assert first == second, field.name


@pytest.mark.parametrize("steps", [2, 3])
def test_normal_batches_are_byte_exact_and_do_not_call_std(steps: int) -> None:
    records = [_record("a", steps=steps), _record("b", steps=steps)]
    _assert_same_batch(collate_object_event_v4(records), original_collate(records))
    events = torch.from_numpy(np.stack([record["event_v4_common_roi"] for record in records]))

    def forbidden(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("proven fast path called std")

    assert not _all_event_rows_constant(events, _std_for_test=forbidden)


def test_fallback_calls_original_std_for_constant_subnormal_and_tiny() -> None:
    shape = (1, 3, 12, 4, 4)
    calls = 0

    def observed(flat: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return flat.std(dim=1)

    cases = []
    constant = torch.zeros(shape, dtype=torch.float32)
    cases.append(constant)
    subnormal = constant.clone()
    subnormal.reshape(-1)[0] = torch.nextafter(
        torch.tensor(0.0), torch.tensor(1.0)
    )
    cases.append(subnormal)
    tiny = constant.clone()
    tiny.reshape(-1)[0] = torch.finfo(torch.float32).tiny
    cases.append(tiny)
    for events in cases:
        expected = bool((events.flatten(0, 1).flatten(1).std(dim=1) <= 0).all())
        assert _all_event_rows_constant(events, _std_for_test=observed) == expected
    assert calls == len(cases)


def test_first_representable_range_above_threshold_skips_std() -> None:
    count = 192
    threshold = torch.tensor(
        4.0
        * torch.finfo(torch.float32).tiny
        * math.sqrt(2.0 * (count - 1)),
        dtype=torch.float32,
    )
    above = torch.nextafter(threshold, torch.tensor(torch.inf, dtype=torch.float32))
    events = torch.zeros((1, 1, 1, 1, count), dtype=torch.float32)
    events.reshape(-1)[-1] = above
    assert events.flatten(0, 1).flatten(1).std(dim=1).item() > 0.0

    def forbidden(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("first FP32 range above threshold called std")

    assert not _all_event_rows_constant(events, _std_for_test=forbidden)


def test_mixed_rows_use_any_fast_path_and_tiny_nonfirst_row_falls_back() -> None:
    normal = torch.zeros((2, 1, 1, 1, 192), dtype=torch.float32)
    normal[1].reshape(-1)[-1] = 1.0

    def forbidden(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("variable nonfirst row called std")

    assert not _all_event_rows_constant(normal, _std_for_test=forbidden)

    tiny = torch.zeros_like(normal)
    tiny[1].reshape(-1)[-1] = torch.finfo(torch.float32).tiny
    called = False

    def observed(flat: torch.Tensor) -> torch.Tensor:
        nonlocal called
        called = True
        return flat.std(dim=1)

    expected = bool((tiny.flatten(0, 1).flatten(1).std(dim=1) <= 0).all())
    assert _all_event_rows_constant(tiny, _std_for_test=observed) == expected
    assert called


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_errors_are_unchanged(value: float) -> None:
    records = [_record("a")]
    changed = records[0]["event_v4_common_roi"].copy()
    changed.reshape(-1)[0] = value
    records[0]["event_v4_common_roi"] = changed
    with pytest.raises(ValueError, match="V4 events contain non-finite values"):
        original_collate(records)
    with pytest.raises(ValueError, match="V4 events contain non-finite values"):
        collate_object_event_v4(records)


def test_constant_and_finite_extreme_decisions_match_original() -> None:
    constant = [_record("constant")]
    constant[0]["event_v4_common_roi"] = np.zeros((3, 12, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Every v4 event channel is constant"):
        original_collate(constant)
    with pytest.raises(ValueError, match="Every v4 event channel is constant"):
        collate_object_event_v4(constant)

    extreme = [_record("extreme")]
    values = extreme[0]["event_v4_common_roi"]
    values.reshape(-1)[::2] = np.finfo(np.float32).max
    values.reshape(-1)[1::2] = -np.finfo(np.float32).max
    _assert_same_batch(collate_object_event_v4(extreme), original_collate(extreme))


def test_other_torch_version_uses_original_std(monkeypatch: pytest.MonkeyPatch) -> None:
    events = torch.arange(3 * 12 * 4 * 4, dtype=torch.float32).reshape(1, 3, 12, 4, 4)
    called = False

    def observed(flat: torch.Tensor) -> torch.Tensor:
        nonlocal called
        called = True
        return flat.std(dim=1)

    monkeypatch.setattr(torch, "__version__", "2.12.0")
    assert not _all_event_rows_constant(events, _std_for_test=observed)
    assert called


def test_collator_ast_diff_is_only_constant_check_delegation() -> None:
    candidate = inspect.getsource(collate_object_event_v4)
    delegated = (
        "    if _all_event_rows_constant(events):\n"
        '        raise ValueError("Every v4 event channel is constant")\n'
    )
    original = (
        "    channel_std = events.flatten(0, 1).flatten(1).std(dim=1)\n"
        "    if bool((channel_std <= 0).all()):\n"
        '        raise ValueError("Every v4 event channel is constant")\n'
    )
    assert candidate.count(delegated) == 1
    reconstructed = candidate.replace(delegated, original)
    assert ast.dump(ast.parse(reconstructed), include_attributes=False) == ast.dump(
        ast.parse(inspect.getsource(original_collate)), include_attributes=False
    )


def test_process_loader_copy_changes_only_collator_import() -> None:
    root = Path(__file__).parents[2]
    base = (root / "operational/train40_system/process_inputs.py").read_text(encoding="utf-8")
    candidate = (
        root / "operational/train40_system/process_inputs_fast_collate.py"
    ).read_text(encoding="utf-8")
    old = "from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4"
    new = "from operational.train40_system.fast_collate import collate_object_event_v4"
    assert candidate.count(new) == 1
    assert candidate.replace(new, old) == base
