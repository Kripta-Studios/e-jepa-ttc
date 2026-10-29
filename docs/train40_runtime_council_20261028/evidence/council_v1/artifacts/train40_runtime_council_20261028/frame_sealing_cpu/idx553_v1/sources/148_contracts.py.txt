"""Immutable RGB-PORT split, frame, timing and batch contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Literal

SPLIT_SALT: Final = "rgb-port-v13-phv-20261008"
Role = Literal["P", "H", "V"]
ROLE_COUNTS: Final = {"P": 24, "H": 8, "V": 8}
LOOKBACK_US: Final = 650_000
MAX_OBSERVATIONS: Final = 8
RGB_SIZE: Final = 128


def assign_roles(groups: list[str], *, salt: str = SPLIT_SALT) -> dict[str, Role]:
    """Assign 24/8/8 sequence groups without consulting targets."""
    unique = sorted(set(map(str, groups)))
    if len(unique) != 40:
        raise ValueError(f"RGB-PORT requires the frozen TRAIN40 groups, got {len(unique)}")
    ranked = sorted(
        unique, key=lambda group: hashlib.sha256(f"{salt}\0{group}".encode()).hexdigest()
    )
    result: dict[str, Role] = {}
    for index, group in enumerate(ranked):
        result[group] = "P" if index < 24 else "H" if index < 32 else "V"
    return result


def assignment_sha256(roles: dict[str, Role]) -> str:
    payload = json.dumps(roles, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RGBFrame:
    """One real public RGB frame with separate capture and availability clocks."""

    frame_id: str
    timestamp_us: int
    available_us: int
    shard_path: str
    member_path: str
    box_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.frame_id or not self.shard_path or not self.member_path:
            raise ValueError("frame identity and public archive paths are required")
        if type(self.timestamp_us) is not int or type(self.available_us) is not int:
            raise TypeError("frame clocks must be Python integer microseconds")
        if self.available_us < self.timestamp_us:
            raise ValueError("availability cannot precede capture")
        x0, y0, x1, y1 = self.box_xyxy
        if x1 <= x0 or y1 <= y0:
            raise ValueError("frame box must have positive extent")


def validate_role_sets(roles: dict[str, list[str]]) -> None:
    """Reject empty or intersecting P/H/V group populations."""
    if set(roles) != {"P", "H", "V"} or any(not roles[name] for name in roles):
        raise ValueError("P/H/V must all be nonempty")
    sets = {name: set(values) for name, values in roles.items()}
    if sets["P"] & sets["H"] or sets["P"] & sets["V"] or sets["H"] & sets["V"]:
        raise ValueError("P/H/V sequence groups overlap")


__all__ = [
    "LOOKBACK_US",
    "MAX_OBSERVATIONS",
    "RGB_SIZE",
    "ROLE_COUNTS",
    "SPLIT_SALT",
    "RGBFrame",
    "Role",
    "assign_roles",
    "assignment_sha256",
    "validate_role_sets",
]
