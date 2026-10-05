"""Lossless original NPZ payloads in both levels of the fixed native input cache."""

from __future__ import annotations

import io
from pathlib import Path

from .common import atomic_bytes
from .exclusive_cache import ExclusiveInputCache


class CompressedInputCache(ExclusiveInputCache):
    """Spend the unchanged6GiB RAM budget on compressed bytes rather than decoded duplicates."""

    @staticmethod
    def unpack(payload: bytes) -> tuple:
        """Recreate identical contiguous FP32 tensors and the original target scalar."""
        import numpy as np
        import torch

        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            return (
                torch.from_numpy(z["events"].copy()),
                torch.from_numpy(z["visible"].copy()),
                float(z["target"]),
            )

    def remember_payload(self, token: str, payload: bytes, *, on_disk: bool) -> None:
        """Prefer exclusive RAM contents while charging every compressed byte."""
        if token in self.values:
            self.values.move_to_end(token)
            return
        size = len(payload)
        if size > self.memory_limit:
            return
        while self.bytes + size > self.memory_limit and self.values:
            victim = next((t for t in self.values if self.token_keys[t] in self.files), None)
            if victim is None:
                if on_disk:
                    return
                victim = next(iter(self.values))
            self.bytes -= len(self.values.pop(victim))
        self.values[token] = payload
        self.bytes += size

    def evict(self, incoming_bytes: int) -> None:
        """Transfer original compressed bytes into RAM before deleting owned disk victims."""
        while self.disk + incoming_bytes > self.limit and self.files:
            name, size = self.files.popitem(last=False)
            candidate = (self.root / name).resolve()
            if candidate.parent != self.root.resolve():
                raise ValueError("compressed cache eviction escaped owned root")
            token = self.key_tokens.get(name)
            if token is not None:
                payload = self.values.get(token)
                if payload is None:
                    payload = candidate.read_bytes()
                self.remember_payload(token, payload, on_disk=False)
                if token in self.values:
                    self.preserved_disk_evictions += 1
            candidate.unlink()
            self.disk -= size

    def get(self, token: str) -> tuple:
        """Change storage only; keep the original encoder, keys, target and tensor arithmetic."""
        import numpy as np
        from e_jepa_ttc.efficient_context.garl_input import encode_record

        raw_stat = (self.c.raw / self.rows[token]["sequence_id"] / "events.h5").stat()
        sequence = self.rows[token]["sequence_id"]
        if (raw_stat.st_size, raw_stat.st_mtime_ns) != self.sequence_stats[sequence]:
            raise ValueError("native raw identity changed during the compressed cache lifetime")
        if token in self.values:
            self.values.move_to_end(token)
            self.hits += 1
            return self.unpack(self.values[token])
        key = self.token_keys[token]
        path: Path = self.root / key
        if path.exists():
            payload = path.read_bytes()
            self.files.move_to_end(key)
            self.hits += 1
            self.remember_payload(token, payload, on_disk=True)
            return self.unpack(payload)
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
        self.remember_payload(token, payload, on_disk=key in self.files)
        return value
