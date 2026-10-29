"""Explicit, lossless runtime overlays for the unchanged V18 primary recipe."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from typing import Any

import numpy as np
import psutil
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC

from . import train
from .data import SensorDataset

ORIGINAL_BACKWARD = train.backward


class PersistentSensors(SensorDataset):
    """Keep checked archives and a bounded decoded-row LRU across GPU scopes.

    Only CPU input bytes are cached. Neural features are always recomputed with
    autograd. File identity changes fail closed even on RAM-cache hits.
    """

    def __init__(self, *, limit_bytes: int = 8 * 1024**3) -> None:
        super().__init__("P")
        self.limit_bytes = limit_bytes
        self.rows_cached: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.retained_bytes = 0
        self.fingerprints: dict[int, tuple] = {}
        self.hits = self.misses = self.peak_bytes = 0
        self.executor: ThreadPoolExecutor | None = None
        self.future: Any = None
        self.shadow: torch.Generator | None = None
        self.pinned = False

    def _fingerprint(self, number: int) -> tuple:
        path = self.folder / f"pack_{number:05d}.zip"
        stat = path.stat()
        receipt = path.with_suffix(".json").stat()
        return (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            receipt.st_size,
            receipt.st_mtime_ns,
            receipt.st_ctime_ns,
        )

    def _row(self, index: int) -> dict[str, np.ndarray]:
        number = index // 256
        fingerprint = self._fingerprint(number)
        prior = self.fingerprints.get(number)
        if prior is not None and prior != fingerprint:
            raise ValueError("Previously admitted sensor pack or receipt changed")
        if index in self.rows_cached:
            self.hits += 1
            self.rows_cached.move_to_end(index)
            return self.rows_cached[index]
        self.misses += 1
        # The parent keeps only sixteen handles; retaining checked files in this
        # layer avoids repeated central-directory parsing without decoding shards.
        import io
        import zipfile

        from operational.range_transfer.common import read, sha

        path = self.folder / f"pack_{number:05d}.zip"
        if number not in self.verified:
            if sha(path) != read(path.with_suffix(".json"))["sha256"]:
                raise ValueError("Sensor pack failed first-use SHA admission")
            if self._fingerprint(number) != fingerprint:
                raise ValueError("Sensor pack changed during admission")
            self.verified.add(number)
            self.fingerprints[number] = fingerprint
        if number not in self.packs:
            self.packs[number] = zipfile.ZipFile(path)
        with np.load(
            io.BytesIO(self.packs[number].read(f"{index:06d}.npz")), allow_pickle=False
        ) as archive:
            row = {key: archive[key] for key in archive.files}
        if str(row["token"]) != str(self.rows.iloc[index].sample_token):
            raise ValueError("Sensor token mismatch")
        size = sum(value.nbytes for value in row.values())
        available = psutil.virtual_memory().available
        while self.rows_cached and (
            self.retained_bytes + size > self.limit_bytes or available < 4 * 1024**3
        ):
            _, old = self.rows_cached.popitem(last=False)
            released = sum(value.nbytes for value in old.values())
            self.retained_bytes -= released
            available = psutil.virtual_memory().available
        if size <= self.limit_bytes and available >= 4 * 1024**3:
            self.rows_cached[index] = row
            self.retained_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
        return row

    def load(self, ids: list[int]) -> ObjectEventV4Batch:
        """Collate exact FP32 rows and optionally pin the completed batch."""
        batch = super().batch(ids)
        if self.pinned:
            for field in fields(batch):
                value = getattr(batch, field.name)
                if isinstance(value, torch.Tensor):
                    setattr(batch, field.name, value.pin_memory())
        return batch

    def _submit(self) -> None:
        assert self.shadow is not None and self.executor is not None
        ids = torch.randint(self.population, (32,), generator=self.shadow).tolist()
        self.future = (ids, self.executor.submit(self.load, ids))

    def prepare(self, sampler_state: torch.Tensor, *, pinned: bool = True) -> None:
        """Forecast on a private generator; the trainer owns the real sampler."""
        self.close()
        self.pinned = pinned
        self.shadow = torch.Generator().set_state(sampler_state)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v18-fast-input")
        self._submit()

    def batch(self, indices: list[int]) -> ObjectEventV4Batch:
        if self.future is None:
            return self.load(indices)
        ids, future = self.future
        if ids != indices:
            raise ValueError("Speculative batch differs from authoritative sampler")
        result = future.result()
        self._submit()
        return result

    def close(self) -> None:
        """Join unconsumed speculative input; keep immutable caches between scopes."""
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.executor = self.future = self.shadow = None

    def shutdown(self) -> None:
        """Release RAM and handles at process exit, outside an optimizer boundary."""
        self.close()
        self.rows_cached.clear()
        self.retained_bytes = 0
        super().close()

    def counters(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "retained_bytes": self.retained_bytes,
            "peak_bytes": self.peak_bytes,
            "verified_packs": len(self.verified),
            "open_packs": len(self.packs),
        }


def batched_backward(
    model: torch.nn.Module,
    batches: list,
    weights: torch.Tensor | None,
    population: int,
    *,
    recompute: bool = False,
) -> dict:
    """Keep the exact native objective; materialize its metrics in one transfer."""
    if not isinstance(model, CausalScaleTTC):
        return ORIGINAL_BACKWARD(model, batches, weights, population, recompute=recompute)
    _, cfg = train.native_config()
    loss, components = train.effective_native_loss(model, batches, cfg, recompute=recompute)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite native objective")
    loss.backward()
    names = ["loss", *components]
    values = torch.stack([loss.detach(), *components.values()]).cpu().tolist()
    return dict(zip(names, values, strict=True))


def install_graph(model: torch.nn.Module) -> None:
    """Compile only the forward, retaining original parameter/state-dict identities."""
    original = model.forward
    model._v18_eager_forward = original
    model.forward = torch.compile(  # type: ignore[method-assign]
        original, backend="cudagraphs", dynamic=False, fullgraph=False
    )
