"""RAW RGB histories whose only pixel storage is authenticated immutable bytes.

This CPU adapter shares arrival-time frame tokens across updates. It does not
upload RAW frames, alter PIL cropping, or cache temporal contexts or TTC outputs.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from operational.train40_runtime.multiobject import RawRGBHistory
from operational.train40_runtime.rgb_frozen_identity import bind_frozen_rgb_history
from operational.train40_runtime.rgb_raw_ring import FrozenRGBFrame, validate_frozen_rgb_frame


class _FrameViews(Mapping[int, np.ndarray]):
    """Return fresh view metadata for each access to owned immutable pixels."""

    __slots__ = ("__tokens",)

    def __init__(self, tokens: Mapping[int, FrozenRGBFrame]) -> None:
        self.__tokens = tokens

    def __getitem__(self, index: int) -> np.ndarray:
        return self.__tokens[index].array

    def __iter__(self) -> Iterator[int]:
        return iter(self.__tokens)

    def __len__(self) -> int:
        return len(self.__tokens)


@dataclass(frozen=True, init=False)
class FrozenRGBHistory(RawRGBHistory):
    """Build all pixels and clocks from factory-created tokens, never other arrays.

    The mapping and triplets are owned snapshots. A caller may mutate a borrowed
    NumPy view's shape without changing any later view or the cache identity.
    Payloads retain their tokens even after the acquisition ring evicts them.
    """

    tokens: Mapping[int, FrozenRGBFrame]

    def __init_subclass__(cls) -> None:
        raise TypeError("FrozenRGBHistory cannot override its token-derived pixel contract")

    def __init__(
        self,
        query_us: int,
        tokens: Mapping[int, FrozenRGBFrame],
        triplets: Sequence[Sequence[int]],
        unavailable_reason: str = "",
    ) -> None:
        if isinstance(query_us, bool) or not isinstance(query_us, int) or query_us < 0:
            raise ValueError("query_us must be a nonnegative integer")
        groups = tuple(tuple(group) for group in triplets)
        owned = dict(tokens)
        required = {index for group in groups for index in group}
        if set(owned) != required:
            raise ValueError("Immutable history tokens must exactly match referenced frame IDs")
        if unavailable_reason and groups:
            raise ValueError("Unavailable immutable history cannot contain RGB observations")
        for index, token in owned.items():
            validate_frozen_rgb_frame(token)
            if isinstance(index, bool) or not isinstance(index, int) or index != token.frame_id:
                raise ValueError("Immutable history key differs from its token frame ID")
        if groups:
            bind_frozen_rgb_history(owned, groups, query_us=query_us)
        clocks = {index: token.timestamp_us for index, token in owned.items()}
        frozen = MappingProxyType(owned)
        super().__init__(
            query_us, _FrameViews(frozen), MappingProxyType(clocks), groups, unavailable_reason
        )
        object.__setattr__(self, "tokens", frozen)
