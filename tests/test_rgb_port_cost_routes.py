"""Zero-optimizer tests for honest RGB-PORT cost route boundaries."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from operational.rgb_port.cost_routes import (
    RotatingBatchOne,
    RouteRuntime,
    _event_histories,
    _select_queries,
    sha256_file_token,
)
from operational.rgb_port.profile import profile_callable


def test_batch_one_profile_rotates_fixed_label_blind_queries() -> None:
    calls: list[int] = []

    def route(index: int) -> torch.Tensor:
        calls.append(index)
        return torch.tensor([index])

    profile = profile_callable(
        RotatingBatchOne(route, list(range(8))),
        scope="HEAD_ONLY:test",
        device="cpu",
        warm_iterations=20,
    )
    assert profile["batch_size"] == 1 and profile["warm_iterations"] == 20
    assert calls == [index % 8 for index in range(21)]


def test_fixed_eight_selection_uses_only_identity_and_route_availability() -> None:
    rows = [{"sample_token": f"s{index}", "target_ttc": index * 1000} for index in range(12)]
    tokens = np.asarray([row["sample_token"] for row in rows])
    caches = {
        "R_CTX": {"sample_token": tokens},
        "F_TRUE": {"sample_token": tokens},
    }
    selected = _select_queries(rows, caches)
    expected = [
        index
        for _, index in sorted((sha256_file_token(f"s{index}"), index) for index in range(12))[:8]
    ]
    assert selected == expected


def test_event_route_uses_frozen_native_history_mapping(tmp_path) -> None:
    rows = [{"sample_token": f"q{index}"} for index in range(5)]
    observations = np.full((5, 8), "", dtype="U2")
    valid = np.zeros((5, 8), dtype=np.bool_)
    observations[0, -1] = "q0"
    valid[0, -1] = True
    observations[4, -4:] = ["q1", "q2", "q3", "q4"]
    valid[4, -4:] = True
    for index in range(1, 4):
        observations[index, -2:] = [f"q{index - 1}", f"q{index}"]
        valid[index, -2:] = True
    path = tmp_path / "history.npz"
    np.savez(
        path,
        query_id=np.asarray([f"q{i}" for i in range(5)]),
        observation_id=observations,
        valid=valid,
    )
    histories = _event_histories(path, rows)
    assert histories[0] == [0]
    assert histories[4] == [1, 2, 3, 4]
    assert max(map(len, histories)) == 4


class _EventContext(nn.Module):
    def forward(self, *args):
        del args
        return {"point_phase": torch.tensor([-0.2])}


class _NeverFusion(nn.Module):
    def forward(self, *args):
        del args
        raise AssertionError("fusion/RGB head ran for a missing-RGB route")


def test_full_fusion_missing_rgb_invokes_exact_event_context_without_rgb_encoder() -> None:
    runtime = object.__new__(RouteRuntime)
    runtime.device = torch.device("cpu")
    runtime.heads = {"E_CTX_MATCHED": _EventContext(), "F_TRUE": _NeverFusion()}
    runtime.event_context = lambda index: (
        torch.zeros(1, 1, 17),
        torch.zeros(1, 1, 4),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[-0.3, -0.2, -0.1]]),
    )
    rgb_calls: list[int] = []

    def missing(index: int):
        rgb_calls.append(index)
        return None

    runtime.rgb_context = missing
    result = RouteRuntime.full_fusion(runtime, 4, "F_TRUE")
    assert result.item() < 0
    assert rgb_calls == [4]
