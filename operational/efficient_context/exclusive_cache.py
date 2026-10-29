"""Lossless two-level TRAIN input cache; preserve RAM entries evicted from disk."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from .common import Campaign, atomic_bytes, digest
from .garl_train import InputCache


class ExclusiveInputCache(InputCache):
    """Retain disk victims in bounded RAM before admitting duplicate resident entries."""

    def __init__(self, c: Campaign, rows: dict) -> None:
        binding = digest(c.out / "garl/PROTOCOL.json")
        stats = {}
        self.token_keys = {}
        self.key_tokens = {}
        for token, row in rows.items():
            sequence = row["sequence_id"]
            if sequence not in stats:
                raw = (c.raw / sequence / "events.h5").stat()
                stats[sequence] = raw.st_size, raw.st_mtime_ns
            size, mtime = stats[sequence]
            key = hashlib.sha256(f"{binding}:{token}:{size}:{mtime}".encode()).hexdigest() + ".npz"
            self.token_keys[token] = key
            self.key_tokens[key] = token
        self.preserved_disk_evictions = 0
        self.sequence_stats = stats
        super().__init__(c, rows)

    @staticmethod
    def decode(path: Path) -> tuple:
        """Decode exactly the original FP32 arrays and scalar target."""
        import numpy as np
        import torch

        with np.load(path, allow_pickle=False) as z:
            return (
                torch.from_numpy(z["events"].copy()),
                torch.from_numpy(z["visible"].copy()),
                float(z["target"]),
            )

    @staticmethod
    def tensor_bytes(value: tuple) -> int:
        """Keep the original accounting convention for resident FP32 tensors."""
        return value[0].numel() * 4 + value[1].numel() * 4

    def remember(self, token: str, value: tuple, *, on_disk: bool) -> None:
        """Prefer evicting duplicates; never evict a RAM-only entry for a duplicate."""
        if token in self.values:
            self.values.move_to_end(token)
            return
        size = self.tensor_bytes(value)
        if size > self.memory_limit:
            return
        while self.bytes + size > self.memory_limit and self.values:
            victim = next((t for t in self.values if self.token_keys[t] in self.files), None)
            if victim is None:
                if on_disk:
                    return
                victim = next(iter(self.values))
            previous = self.values.pop(victim)
            self.bytes -= self.tensor_bytes(previous)
        self.values[token] = value
        self.bytes += size

    def evict(self, incoming_bytes: int) -> None:
        """Transfer recognized disk victims to RAM under the unchanged capacity limits."""
        while self.disk + incoming_bytes > self.limit and self.files:
            name, size = self.files.popitem(last=False)
            candidate = (self.root / name).resolve()
            if candidate.parent != self.root.resolve():
                raise ValueError("cache eviction escaped owned root")
            token = self.key_tokens.get(name)
            if token is not None:
                value = self.values[token] if token in self.values else self.decode(candidate)
                self.remember(token, value, on_disk=False)
                if token in self.values:
                    self.preserved_disk_evictions += 1
            candidate.unlink()
            self.disk -= size

    def get(self, token: str) -> tuple:
        """Same original encoder, FP32 npz format and key; only storage placement differs."""
        import numpy as np

        from e_jepa_ttc.efficient_context.garl_input import encode_record

        sequence = self.rows[token]["sequence_id"]
        raw_stat = (self.c.raw / sequence / "events.h5").stat()
        if (raw_stat.st_size, raw_stat.st_mtime_ns) != self.sequence_stats[sequence]:
            raise ValueError("native raw identity changed during the cache lifetime")
        if token in self.values:
            self.values.move_to_end(token)
            self.hits += 1
            return self.values[token]
        key = self.token_keys[token]
        path = self.root / key
        if path.exists():
            value = self.decode(path)
            self.files.move_to_end(key)
            self.hits += 1
            self.remember(token, value, on_disk=True)
            return value
        self.c.require_resources()
        value = encode_record(self.rows[token], self.pool, self.c.raw)
        self.reads += 1
        stream = io.BytesIO()
        np.savez_compressed(
            stream, events=value[0].numpy(), visible=value[1].numpy(), target=value[2]
        )
        payload = stream.getvalue()
        self.evict(len(payload))
        if len(payload) <= self.limit:
            atomic_bytes(path, payload)
            self.files[key] = len(payload)
            self.disk += len(payload)
        self.remember(token, value, on_disk=key in self.files)
        return value
