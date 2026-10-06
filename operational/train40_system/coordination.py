"""Bounded transfer overlap and equivalent diagnostics for the frozen TRAIN40 recipe."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch


def foreground_mask(
    boxes: torch.Tensor, source_height: int, source_width: int, feat_h: int, feat_w: int
) -> torch.Tensor:
    """Rasterize the original floor/ceil, clamped, half-open rectangles without scalar copies."""
    if min(source_height, source_width, feat_h, feat_w) <= 0:
        raise ValueError("relational fg/bg diagnostic dimensions must be positive")
    indices = [0, 1] if boxes.shape[1] == 2 else [1, 2]
    result = torch.zeros((len(boxes), 2, feat_h, feat_w), dtype=torch.bool, device=boxes.device)
    present = [(ep, index) for ep, index in enumerate(indices) if index < boxes.shape[1]]
    if not present:
        return result
    selected = boxes[:, [index for _, index in present]].float()
    if bool(torch.isnan(selected).any()):
        raise ValueError("cannot convert float NaN to integer")
    x1 = torch.floor(selected[..., 0] * (feat_w / float(source_width))).clamp(0, feat_w)
    y1 = torch.floor(selected[..., 1] * (feat_h / float(source_height))).clamp(0, feat_h)
    x2 = torch.ceil(selected[..., 2] * (feat_w / float(source_width))).clamp(0, feat_w)
    y2 = torch.ceil(selected[..., 3] * (feat_h / float(source_height))).clamp(0, feat_h)
    x = torch.arange(feat_w, device=boxes.device)[None, None, None, :]
    y = torch.arange(feat_h, device=boxes.device)[None, None, :, None]
    inside = (
        (x >= x1[..., None, None])
        & (x < x2[..., None, None])
        & (y >= y1[..., None, None])
        & (y < y2[..., None, None])
    )
    result[:, [ep for ep, _ in present]] = inside
    return result


def relational_diagnostic(
    student_features: torch.Tensor,
    teacher_relations: torch.Tensor,
    teacher_valid: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    source_height: int,
    source_width: int,
    feat_h: int,
    feat_w: int,
    components: dict[str, torch.Tensor],
) -> None:
    """Retain every diagnostic and reduction; replace only the scalar rectangle construction."""
    from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps

    with torch.no_grad():
        student_rels = local_cosine_relation_maps(student_features)
        combined_valid = teacher_valid.bool() & student_rels.valid
        error_map = (student_rels.values - teacher_relations.float()).abs() * combined_valid.float()
        fg_mask = foreground_mask(boxes_xyxy, source_height, source_width, feat_h, feat_w)
        fg_expanded = fg_mask.unsqueeze(2).expand_as(error_map)
        valid_fg = combined_valid & fg_expanded
        valid_bg = combined_valid & ~fg_expanded
        fg_loss = (
            error_map[valid_fg].mean() if valid_fg.any() else error_map.new_tensor(float("nan"))
        )
        bg_loss = (
            error_map[valid_bg].mean() if valid_bg.any() else error_map.new_tensor(float("nan"))
        )
        fg_fraction = (
            valid_fg.float().sum() / combined_valid.float().sum()
            if combined_valid.any()
            else error_map.new_tensor(float("nan"))
        )
        components["dinov3_relational_fg_loss"] = fg_loss
        components["dinov3_relational_bg_loss"] = bg_loss
        components["dinov3_relational_fg_fraction"] = fg_fraction


def scalar_metrics(components: dict[str, torch.Tensor], total: torch.Tensor) -> dict[str, float]:
    """Copy the identical FP32 scalars to CPU in one transfer, retaining insertion order."""
    names = [*components, "total"]
    values = torch.stack([value.detach().float() for value in (*components.values(), total)])
    return dict(zip(names, values.cpu().tolist(), strict=True))


class BatchSource(Protocol):
    """Immutable sensor and teacher source consumed in sampler order."""

    def batch(self, ids: list[int]) -> ObjectEventV4Batch: ...


@dataclass
class DeviceTicket:
    """Own pinned storage until the asynchronous copy has completed."""

    batch: ObjectEventV4Batch
    host: ObjectEventV4Batch | None
    ready: torch.cuda.Event
    ids: tuple[int, ...]

    def consume(self) -> ObjectEventV4Batch:
        """Order model reads after this copy without synchronizing unrelated streams."""
        self.ready.synchronize()
        stream = torch.cuda.current_stream()
        # PyTorch 2.11 stubs confuse the CUDA Event subclass with the generic C Event.
        stream.wait_event(self.ready)  # pyright: ignore[reportArgumentType]
        for field in fields(self.batch):
            tensor = getattr(self.batch, field.name)
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                tensor.record_stream(stream)
        self.host = None
        return self.batch


class DevicePrefetch:
    """Keep at most three deterministic batches ahead using one collator and one copy stream."""

    def __init__(self, source: BatchSource, *, depth: int = 3) -> None:
        if depth < 1 or depth > 3:
            raise ValueError("Bounded prefetch depth must be in [1,3]")
        self.source, self.depth = source, depth
        self.stream = torch.cuda.Stream()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: deque[Future[DeviceTicket]] = deque()
        self.order: torch.Tensor | None = None
        self.scheduled = self.expected = 0
        self.failure: BaseException | None = None
        self.closed = False

    def _copy(self, ids: list[int]) -> DeviceTicket:
        host = self.source.batch(ids)
        pinned = replace(
            host,
            **{
                field.name: tensor.pin_memory()
                for field in fields(host)
                if isinstance(tensor := getattr(host, field.name), torch.Tensor)
            },
        )
        with torch.cuda.stream(self.stream):
            batch = pinned.to(torch.device("cuda"), non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        return DeviceTicket(batch, pinned, ready, tuple(ids))

    def _shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        try:
            self.stream.synchronize()
        finally:
            self.pending.clear()

    def _require_available(self) -> None:
        if self.failure is not None:
            raise RuntimeError("Device prefetch previously failed") from self.failure
        if self.closed:
            raise RuntimeError("Device prefetch is closed")

    def take(self, order: torch.Tensor, start: int, batch_size: int) -> DeviceTicket:
        """Submit only the current epoch and reject skipped or reordered consumption."""
        self._require_available()
        if order.ndim != 1:
            raise ValueError("Prefetch order must be a one-dimensional tensor")
        if batch_size <= 0:
            raise ValueError("Prefetch batch_size must be positive")
        if start < 0 or start > len(order):
            raise ValueError("Prefetch start must lie within the epoch order")
        if order is not self.order:
            if self.pending:
                raise ValueError("Previous epoch still has unconsumed batches")
            if self.order is not None and (
                self.expected != len(self.order) or self.scheduled != len(self.order)
            ):
                raise ValueError("Previous epoch was not completely consumed")
            if self.order is not None and start != 0:
                raise ValueError("A newly admitted epoch must start at cursor zero")
            self.order, self.scheduled, self.expected = order, start, start
        if start != self.expected:
            raise ValueError("Prefetch consumption differs from checkpoint sampler cursor")
        while len(self.pending) < self.depth and self.scheduled < len(order):
            ids = order[self.scheduled : self.scheduled + batch_size].tolist()
            self.pending.append(self.executor.submit(self._copy, ids))
            self.scheduled += len(ids)
        if not self.pending:
            raise ValueError("No batch remains in the admitted epoch")
        future = self.pending.popleft()
        try:
            ticket = future.result()
        except BaseException as error:
            self.failure = error
            self.closed = True
            self._shutdown()
            raise RuntimeError("Device prefetch worker failed") from error
        if len(ticket.batch.events) != len(ticket.ids):
            error = ValueError(
                "Prefetch source returned a different batch size than the admitted sampler IDs"
            )
            self.failure = error
            self.closed = True
            self._shutdown()
            raise error
        self.expected += len(ticket.ids)
        return ticket

    def close(self) -> None:
        """Finish outstanding copies before releasing their pinned source buffers."""
        if self.closed:
            return
        self.closed = True
        self._shutdown()
