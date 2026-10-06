"""CPU-only behavioral tests for the bounded TRAIN40 device prefetcher."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from operational.train40_system.coordination import DevicePrefetch


class FakeEvent:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def record(self, _stream: FakeStream) -> None:
        self.log.append("record")

    def synchronize(self) -> None:
        self.log.append("event_synchronize")


class FakeStream:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.synchronize_calls = 0

    def wait_event(self, _event: FakeEvent) -> None:
        self.log.append("wait_event")

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.log.append("stream_synchronize")


@dataclass
class FakeBatch:
    events: torch.Tensor
    target: torch.Tensor
    tokens: list[str]

    def to(self, _device: torch.device, *, non_blocking: bool) -> FakeBatch:
        assert non_blocking
        return replace(self, events=self.events.clone(), target=self.target.clone())


class FakeSource:
    def __init__(self, *, fail_on: int | None = None, size_offset: int = 0) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.fail_on = fail_on
        self.size_offset = size_offset

    def batch(self, ids: list[int]) -> Any:
        admitted = tuple(ids)
        self.calls.append(admitted)
        if ids and ids[0] == self.fail_on:
            raise OSError("synthetic collator failure")
        size = len(ids) + self.size_offset
        return FakeBatch(torch.arange(size), torch.arange(size), [str(value) for value in ids])


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    current = FakeStream(log)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: FakeStream(log))
    monkeypatch.setattr(torch.cuda, "Event", lambda: FakeEvent(log))
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    return log


def test_prefetch_preserves_order_final_batch_and_copy_lifetime(fake_cuda: list[str]) -> None:
    source = FakeSource()
    pipeline = DevicePrefetch(source, depth=3)
    order = torch.tensor([6, 1, 5, 2, 4, 0, 3])

    first = pipeline.take(order, 0, 3)
    assert first.ids == (6, 1, 5)
    assert len(pipeline.pending) <= 3
    assert first.host is not None
    first.consume()
    assert first.host is None
    assert fake_cuda.index("event_synchronize") < fake_cuda.index("wait_event")

    second = pipeline.take(order, 3, 3)
    final = pipeline.take(order, 6, 3)
    assert second.ids == (2, 4, 0)
    assert final.ids == (3,)
    assert pipeline.expected == len(order)

    next_order = torch.tensor([9, 8])
    assert pipeline.take(next_order, 0, 3).ids == (9, 8)
    pipeline.close()
    pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.take(next_order, 2, 3)


def test_resume_cursor_and_epoch_switch_require_exact_order(fake_cuda: list[str]) -> None:
    source = FakeSource()
    pipeline = DevicePrefetch(source, depth=1)
    order = torch.arange(6)

    assert pipeline.take(order, 2, 2).ids == (2, 3)
    with pytest.raises(ValueError, match="checkpoint sampler cursor"):
        pipeline.take(order, 5, 2)
    with pytest.raises(ValueError, match="not completely consumed"):
        pipeline.take(torch.arange(3), 0, 2)

    assert pipeline.take(order, 4, 2).ids == (4, 5)
    with pytest.raises(ValueError, match="cursor zero"):
        pipeline.take(torch.arange(3), 1, 2)
    pipeline.close()


def test_worker_error_is_sticky_and_does_not_skip_ids(fake_cuda: list[str]) -> None:
    source = FakeSource(fail_on=0)
    pipeline = DevicePrefetch(source, depth=1)
    order = torch.arange(4)

    with pytest.raises(RuntimeError, match="worker failed") as first:
        pipeline.take(order, 0, 2)
    assert isinstance(first.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match="previously failed") as repeated:
        pipeline.take(order, 0, 2)
    assert repeated.value.__cause__ is first.value.__cause__
    assert source.calls == [(0, 1)]
    pipeline.close()


def test_source_cannot_change_admitted_batch_cardinality(fake_cuda: list[str]) -> None:
    pipeline = DevicePrefetch(FakeSource(size_offset=-1), depth=1)
    order = torch.arange(4)

    with pytest.raises(ValueError, match="different batch size"):
        pipeline.take(order, 0, 2)
    with pytest.raises(RuntimeError, match="previously failed"):
        pipeline.take(order, 0, 2)
    pipeline.close()


@pytest.mark.parametrize(
    ("order", "start", "batch_size", "message"),
    [
        (torch.arange(3).reshape(1, 3), 0, 1, "one-dimensional"),
        (torch.arange(3), 0, 0, "positive"),
        (torch.arange(3), -1, 1, "within"),
        (torch.arange(3), 4, 1, "within"),
    ],
)
def test_invalid_admission_never_reaches_source(
    fake_cuda: list[str], order: torch.Tensor, start: int, batch_size: int, message: str
) -> None:
    source = FakeSource()
    pipeline = DevicePrefetch(source)
    with pytest.raises(ValueError, match=message):
        pipeline.take(order, start, batch_size)
    assert source.calls == []
    pipeline.close()
