"""Focused regressions for the public eAP causal-scale runner."""

from __future__ import annotations

import torch

from scripts.train_causal_scale_eap_screen import _reset_peak_memory_stats


def test_peak_memory_reset_selects_cuda_and_uses_default_api(monkeypatch) -> None:
    selected: list[torch.device] = []
    resets = 0

    def record_device(device: torch.device) -> None:
        selected.append(device)

    def record_reset() -> None:
        nonlocal resets
        resets += 1

    monkeypatch.setattr(torch.cuda, "set_device", record_device)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", record_reset)

    device = torch.device("cuda", 0)
    _reset_peak_memory_stats(device)

    assert selected == [device]
    assert resets == 1


def test_peak_memory_reset_is_noop_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("must not select CUDA")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda: (_ for _ in ()).throw(AssertionError("must not reset CUDA")),
    )

    _reset_peak_memory_stats(torch.device("cpu"))
