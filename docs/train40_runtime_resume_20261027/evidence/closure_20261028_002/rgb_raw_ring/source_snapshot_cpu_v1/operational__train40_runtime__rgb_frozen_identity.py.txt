"""Exact RGB history identity from arrival-time immutable frame tokens.

This adapter hashes a bounded metadata descriptor, never RAW payload bytes.
Producer pixels must come from the referenced tokens' ``array`` views. Binding
tokens while cropping the original mutable arrays would violate this contract.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence

from operational.train40_runtime.feature_cache import RGBHistoryIdentity
from operational.train40_runtime.rgb_raw_ring import FrozenRGBFrame, validate_frozen_rgb_frame

_MAX_DESCRIPTOR_BYTES = 16 * 1024


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def bind_frozen_rgb_history(
    frames: Mapping[int, FrozenRGBFrame],
    triplets: Sequence[Sequence[int]],
    *,
    query_us: int,
    max_raw_bytes: int = 128 * 1024**2,
) -> RGBHistoryIdentity:
    """Validate a causal history without rescanning owned immutable RGB bytes.

    Clocks are derived exclusively from factory-authenticated tokens. Each
    referenced mapping key must equal its token's frame ID, and every token
    must share one source name/revision. All producer inputs, including complete
    history, shape and exact deltas, are bound. The current query time is only
    an availability check; context ages and TTC must still be recomputed.

    ``hashed_bytes=0`` counts RAW payload scanning in this call, not arrival
    hashing. The factory's frame-copy/checksum cost belongs at sensor arrival
    inside the stated sensor-to-TTC clock. This function hashes at most 16 KiB
    of metadata and rejects larger descriptors. Use ``token.array`` for all
    subsequent crops; a checksum token cannot attest other mutable arrays.
    """
    started = time.perf_counter()
    _integer(query_us, "query_us")
    _integer(max_raw_bytes, "max_raw_bytes", minimum=1)
    groups = tuple(tuple(_integer(index, "frame index") for index in group) for group in triplets)
    if not 1 <= len(groups) <= 8 or len({len(group) for group in groups}) != 1:
        raise ValueError("RGB history requires one to eight equally sized observations")
    if any(len(group) not in (2, 3) for group in groups):
        raise ValueError("Each RGB observation must contain two or three frames")
    frame_ids = sorted({index for group in groups for index in group})
    if len(frame_ids) > 10:
        raise ValueError("RGB history exceeds ten distinct frames")
    clocks: dict[int, int] = {}
    source: tuple[str, str] | None = None
    descriptors = []
    raw_bytes = 0
    for index in frame_ids:
        if index not in frames:
            raise ValueError("RGB history lacks a referenced immutable frame")
        token = frames[index]
        validate_frozen_rgb_frame(token)
        if token.frame_id != index:
            raise ValueError("RGB mapping key differs from its immutable token frame ID")
        origin = token.source
        current_source = (origin.name, origin.revision_sha256)
        if source is not None and current_source != source:
            raise ValueError("RGB history cannot mix immutable source names/revisions")
        source = current_source
        clocks[index] = token.timestamp_us
        raw_bytes += token.hashed_bytes
        if raw_bytes > max_raw_bytes:
            raise ValueError("Shared immutable RGB history exceeds the configured byte bound")
        descriptors.append((index, token.timestamp_us, token.shape, token.identity_sha256))
    for group in groups:
        times = [clocks[index] for index in group]
        if any(a >= b for a, b in zip(times[:-1], times[1:], strict=True)):
            raise ValueError("RGB frame timestamps must be strictly chronological")
        if times[-1] > query_us or query_us - times[0] > 650_000:
            raise ValueError("RGB history must be available and inside the causal support")
    anchors = [clocks[group[-1]] for group in groups]
    if any(a >= b for a, b in zip(anchors[:-1], anchors[1:], strict=True)):
        raise ValueError("RGB observation anchors must be strictly chronological")
    deltas = tuple(
        tuple(
            ((clocks[b] - clocks[a]) / 1e6).hex()
            for a, b in zip(group[:-1], group[1:], strict=True)
        )
        for group in groups
    )
    assert source is not None
    source_binding = ("IMMUTABLE_OWNED_RGB_V1", *source)
    descriptor = json.dumps(
        (source_binding, groups, descriptors, deltas),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if len(descriptor) > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("Immutable RGB identity metadata exceeds the 16 KiB descriptor bound")
    digest = hashlib.sha256(descriptor).hexdigest()
    return RGBHistoryIdentity(
        digest=digest,
        source_binding=source_binding,
        triplets=groups,
        frame_clocks_us=tuple((index, clocks[index]) for index in frame_ids),
        delta_seconds_hex=deltas,
        raw_bytes=raw_bytes,
        hashed_bytes=0,
        fingerprint_seconds=time.perf_counter() - started,
    )
