"""Focused regression tests for partially visible CUDA runtimes."""

from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.reproducibility import (
    cuda_device_count,
    cuda_device_name,
    cuda_is_usable,
    environment_snapshot,
    resolve_device,
)


def _mask_cuda_without_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model ``is_available=True`` with no addressable visible devices."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def test_auto_uses_cpu_when_cuda_has_no_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mask_cuda_without_allocating(monkeypatch)

    assert resolve_device("auto") == torch.device("cpu")
    assert cuda_device_count() == 0
    assert not cuda_is_usable()
    assert cuda_device_name() is None

    snapshot = environment_snapshot()
    assert snapshot["cuda_available"] is False
    assert snapshot["cuda_runtime_reported_available"] is True
    assert snapshot["cuda_device_count"] == 0
    assert snapshot["gpu_name"] is None


@pytest.mark.parametrize("requested", ("cuda", "cuda:0"))
def test_explicit_cuda_fails_before_any_allocation(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    _mask_cuda_without_allocating(monkeypatch)

    with pytest.raises(RuntimeError, match="no visible CUDA device"):
        resolve_device(requested)


def test_invisible_device_name_is_never_queried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mask_cuda_without_allocating(monkeypatch)
    queried = False

    def fail_if_queried(_: int) -> str:
        nonlocal queried
        queried = True
        raise AssertionError("an invisible CUDA device was queried")

    monkeypatch.setattr(torch.cuda, "get_device_name", fail_if_queried)

    assert cuda_device_name(0) is None
    assert not queried


def test_device_name_returns_none_when_driver_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _: (_ for _ in ()).throw(RuntimeError("driver unavailable")),
    )

    assert cuda_device_name(0) is None


def test_cpu_metadata_does_not_report_a_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert cuda_device_name(torch.device("cpu")) is None


def test_object_device_helper_uses_the_same_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mask_cuda_without_allocating(monkeypatch)

    from e_jepa_ttc.training.object_jepa import _device

    assert _device("auto") == torch.device("cpu")


@pytest.mark.parametrize(
    "module_name",
    (
        "e_jepa_ttc.training.jepa",
        "e_jepa_ttc.training.supervised",
        "e_jepa_ttc.training.prober",
        "e_jepa_ttc.training.object_jepa",
        "e_jepa_ttc.training.carla_jepa",
        "e_jepa_ttc.training.object_geo_trainer",
    ),
)
def test_seed_helpers_skip_cuda_seed_when_no_device_is_visible(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    _mask_cuda_without_allocating(monkeypatch)
    module = __import__(module_name, fromlist=["_set_seed"])
    called = False

    def fail_if_called(_: int) -> None:
        nonlocal called
        called = True
        raise AssertionError("CUDA seed must not be called without a device")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", fail_if_called)
    module._set_seed(7)
    assert not called
