"""Deterministic native input lookahead with CPU-only workers and bounded open readers."""

from __future__ import annotations

import atexit
import io
import os
from collections import OrderedDict, deque
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import TYPE_CHECKING

from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

from .common import Campaign, atomic_bytes
from .compressed_cache import CompressedInputCache

if TYPE_CHECKING:
    import torch
    from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader

_pool: MultiReaderPool | None = None


class MultiReaderPool(ReaderPool):
    """Keep at most eight identical read-only CachedEventReaders in each CPU worker."""

    def __init__(self, limit: int = 8) -> None:
        super().__init__()
        self.limit = limit
        self.readers: OrderedDict = OrderedDict()

    def get(self, path: str | Path) -> CachedEventReader:
        """Reuse an unchanged pinned reader and close the least recently used victim."""
        from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader

        path = Path(path).resolve(strict=True)
        if path in self.readers:
            self.readers.move_to_end(path)
            return self.readers[path]
        if len(self.readers) >= self.limit:
            _, reader = self.readers.popitem(last=False)
            reader.close()
        reader = CachedEventReader(path)
        reader.open()
        self.readers[path] = reader
        return reader

    def close(self) -> None:
        """Release every worker-owned HDF5 handle."""
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()


def worker_init() -> None:
    """Use the original four-thread CPU arithmetic; never initialize CUDA or a model."""
    global _pool
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    _pool = MultiReaderPool()
    atexit.register(_pool.close)


def prepare_record(row: dict, raw_root: str) -> tuple:
    """Execute the original frozen native encoder without parent RNG or output writes."""
    import psutil
    from e_jepa_ttc.efficient_context.garl_input import encode_record

    if psutil.virtual_memory().available < 2 * 1024**3:
        raise InterruptedError("CPU input worker requires at least2GiB host available RAM")
    if _pool is None:
        raise RuntimeError("CPU input worker was not initialized")
    events, visible, target = encode_record(row, _pool, Path(raw_root))
    return events.numpy(), visible.numpy(), target


def sampler_indices(
    order: torch.Tensor, cursor: int, epoch: int, generator: torch.Generator
) -> Iterator[int]:
    """Read a private restored sampler copy; never consume the trainer's generator."""
    import torch

    while epoch <= 50:
        for position in range(cursor, len(order)):
            yield int(order[position])
        epoch += 1
        cursor = 0
        if epoch <= 50:
            order = torch.randperm(len(order), generator=generator)


class ParallelInputCache(CompressedInputCache):
    """Four CPU producers, one cache writer and the unchanged optimizer batch order."""

    def __init__(self, c: Campaign, rows: dict) -> None:
        if c.policy["max_tree_rss_gib"] * 1024**3 < 16_000_000_000:
            raise ValueError("four CPU input workers require the new16GB resource authorization")
        self.executor: ProcessPoolExecutor | None = None
        self.pending: deque[tuple[str, Future | None]] = deque()
        self.upcoming: Iterator[str] = iter(())
        self.active = False
        super().__init__(c, rows)
        self.memory_limit = 8 * 1024**3

    def activate(self, fit: dict) -> None:
        """Plan only the registered fit's exact sampler, including resumed cursor and RNG."""
        import numpy as np
        import torch
        from e_jepa_ttc.simplex_t.training import state_digest

        self.drain()
        with np.load(
            self.c.out / f"garl/admission/{fit['key']}_native_tokens.npz", allow_pickle=False
        ) as z:
            tokens = z["tokens"].astype(str)
        generator = torch.Generator().manual_seed(7)
        order = torch.randperm(len(tokens), generator=generator)
        cursor, epoch = 0, 1
        path = self.c.out / f"garl/fits/{fit['key']}/checkpoint_last.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            seal = state.pop("state_sha256")
            if state_digest(state) != seal or state["protocol_sha256"] != self.binding:
                raise ValueError("parallel input lookahead requires the exact complete checkpoint")
            order, cursor, epoch = state["order"], state["cursor"], state["epoch"]
            generator.set_state(state["sampler_rng"])
            del state
        self.upcoming = (str(tokens[i]) for i in sampler_indices(order, cursor, epoch, generator))
        self.active = epoch <= 50
        if self.active and self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=4, initializer=worker_init)
        self.fill()

    def fill(self) -> None:
        """Bound lookahead to64 decoded records, while keeping cache writes in sampler order."""
        if not self.active:
            return
        while len(self.pending) < 64:
            token = next(self.upcoming, None)
            if token is None:
                self.active = False
                return
            key = self.token_keys[token]
            future = None
            if token not in self.values and not (self.root / key).exists():
                self.c.require_resources()
                if self.executor is None:
                    raise RuntimeError("CPU preparation executor missing")
                future = self.executor.submit(prepare_record, self.rows[token], str(self.c.raw))
            self.pending.append((token, future))

    def get(self, token: str) -> tuple:
        """Return exactly the requested sampler entry; completion order cannot change batching."""
        import numpy as np
        import torch

        future = None
        if self.pending:
            expected, future = self.pending.popleft()
            if expected != token:
                raise ValueError("CPU input lookahead differs from the frozen trainer sampler")
        if future is not None and (
            token in self.values or (self.root / self.token_keys[token]).exists()
        ):
            future.cancel()
            future = None
        if future is None:
            value = super().get(token)
        else:
            self.c.require_resources()
            sequence = self.rows[token]["sequence_id"]
            stat = (self.c.raw / sequence / "events.h5").stat()
            if (stat.st_size, stat.st_mtime_ns) != self.sequence_stats[sequence]:
                raise ValueError("native raw identity changed during CPU preparation")
            try:
                x, visible, target = future.result()
            except BrokenProcessPool as error:
                raise OSError(
                    "CPU input worker stopped; preserve the optimizer boundary"
                ) from error
            value = torch.from_numpy(x), torch.from_numpy(visible), float(target)
            self.reads += 1
            stream = io.BytesIO()
            np.savez_compressed(stream, events=x, visible=visible, target=target)
            payload = stream.getvalue()
            self.evict(len(payload))
            key = self.token_keys[token]
            if len(payload) <= self.limit:
                atomic_bytes(self.root / key, payload)
                self.files[key] = len(payload)
                self.disk += len(payload)
            self.remember_payload(token, payload, on_disk=key in self.files)
        self.fill()
        return value

    def drain(self) -> None:
        """Discard lookahead outputs without changing any scientific state or artifacts."""
        for _, future in self.pending:
            if future is not None:
                future.cancel()
        self.pending.clear()
        self.active = False

    def close(self) -> None:
        """Stop only CPU input workers and release all owned readers and cached payloads."""
        self.drain()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        super().close()
