"""Causal RGB history indexing from published eAP exposure timestamps.

The release exposes sensor exposure start/end timestamps, but no arrival clock.
Accordingly this module proves sensor-time causality only: a past frame must have
finished exposure no later than the current frame starts.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

TemporalControl = Literal[
    "current_only",
    "current_repeated",
    "correct_past",
    "permuted_past",
    "delayed_past",
]
CONTROLS: tuple[TemporalControl, ...] = (
    "current_only",
    "current_repeated",
    "correct_past",
    "permuted_past",
    "delayed_past",
)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer microsecond timestamp")
    return value


@dataclass(frozen=True)
class RGBExposure:
    """One published RGB exposure interval in sensor time."""

    sample_token: str
    sequence_id: str
    exposure_start_us: int
    exposure_end_us: int

    def __post_init__(self) -> None:
        if not self.sample_token or not self.sequence_id:
            raise ValueError("RGB exposure identity must be nonempty")
        start = _integer(self.exposure_start_us, "exposure_start_us")
        end = _integer(self.exposure_end_us, "exposure_end_us")
        if start < 0 or end < start:
            raise ValueError("RGB exposure interval must be nonnegative and ordered")


@dataclass(frozen=True)
class TemporalRGBSample:
    """Current frame plus two causal history slots and explicit validity masks."""

    sequence_id: str
    current_token: str
    current_exposure_start_us: int
    current_exposure_end_us: int
    decision_sensor_us: int
    history_tokens: tuple[str | None, str | None]
    history_mask: tuple[bool, bool]
    history_exposure_end_us: tuple[int | None, int | None]
    history_age_us: tuple[int | None, int | None]
    control: TemporalControl
    availability_scope: str = "sensor_exposure_time_only; arrival_time_unavailable"

    def __post_init__(self) -> None:
        if self.control not in CONTROLS:
            raise ValueError(f"Unknown temporal control: {self.control}")
        if not self.sequence_id or not self.current_token:
            raise ValueError("Temporal sample identity must be nonempty")
        start = _integer(self.current_exposure_start_us, "current_exposure_start_us")
        end_current = _integer(self.current_exposure_end_us, "current_exposure_end_us")
        decision = _integer(self.decision_sensor_us, "decision_sensor_us")
        if start < 0 or end_current < start:
            raise ValueError("Current exposure interval must be nonnegative and ordered")
        if self.decision_sensor_us != self.current_exposure_end_us:
            raise ValueError("Decision sensor time must equal current exposure end")
        fields = (
            self.history_tokens,
            self.history_mask,
            self.history_exposure_end_us,
            self.history_age_us,
        )
        if any(len(value) != 2 for value in fields):
            raise ValueError("Temporal history fields must contain exactly two slots")
        if any(type(value) is not bool for value in self.history_mask):
            raise ValueError("History masks must contain booleans")
        for index, token in enumerate(self.history_tokens):
            valid = self.history_mask[index]
            end_value = self.history_exposure_end_us[index]
            age = self.history_age_us[index]
            if valid != (token is not None and end_value is not None and age is not None):
                raise ValueError("History token, timestamp, age and mask must agree")
            if valid and end_value is not None and age is not None:
                history_end = _integer(end_value, "history_exposure_end_us")
                history_age = _integer(age, "history_age_us")
                if history_age < 0 or history_age != decision - history_end:
                    raise ValueError("History exceeds the sensor-time decision cutoff")
                if token != self.current_token and history_end > start:
                    raise ValueError("Past history overlaps the current exposure")
        if self.control == "current_repeated":
            if (
                self.history_tokens != (self.current_token, self.current_token)
                or self.history_mask != (True, True)
                or self.history_exposure_end_us != (end_current, end_current)
                or self.history_age_us != (0, 0)
            ):
                raise ValueError("Repeated-current control must contain exactly two current frames")
        elif self.current_token in self.history_tokens:
            raise ValueError("Only repeated-current control may reuse the current frame")


class TemporalRGBIndex:
    """Validated sequence-local lookup for current and previous RGB frames."""

    def __init__(self, rows: Iterable[Mapping[str, object]]) -> None:
        sequences: dict[str, list[RGBExposure]] = {}
        tokens: dict[str, RGBExposure] = {}
        previous_start: dict[str, int] = {}
        for row in rows:
            token, sequence = row.get("sample_token"), row.get("sequence_id")
            if not isinstance(token, str) or not isinstance(sequence, str):
                raise ValueError("sample_token and sequence_id must be strings")
            frame = RGBExposure(
                sample_token=token,
                sequence_id=sequence,
                exposure_start_us=_integer(
                    row.get("rgb_exposure_start_timestamp_us"),
                    "rgb_exposure_start_timestamp_us",
                ),
                exposure_end_us=_integer(
                    row.get("rgb_exposure_end_timestamp_us"),
                    "rgb_exposure_end_timestamp_us",
                ),
            )
            if token in tokens:
                raise ValueError(f"Duplicate sample token: {token}")
            if frame.exposure_start_us <= previous_start.get(sequence, -1):
                raise ValueError(
                    f"Late or duplicate exposure start in sequence {sequence}: {token}"
                )
            previous_start[sequence] = frame.exposure_start_us
            tokens[token] = frame
            sequences.setdefault(sequence, []).append(frame)
        if not tokens:
            raise ValueError("Temporal RGB index cannot be empty")
        self._tokens = tokens
        self._sequences = {key: tuple(value) for key, value in sequences.items()}
        self._positions = {
            frame.sample_token: index
            for sequence in self._sequences.values()
            for index, frame in enumerate(sequence)
        }

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        """Return deterministic sequence IDs."""
        return tuple(sorted(self._sequences))

    @property
    def sample_tokens(self) -> tuple[str, ...]:
        """Return tokens in sequence/time order."""
        return tuple(
            frame.sample_token
            for sequence in self.sequence_ids
            for frame in self._sequences[sequence]
        )

    def _causal_past(self, current: RGBExposure, *, limit: int | None = None) -> list[RGBExposure]:
        sequence = self._sequences[current.sequence_id]
        position = self._positions[current.sample_token]
        if limit is not None:
            result = []
            for frame in reversed(sequence[:position]):
                if frame.exposure_end_us <= current.exposure_start_us:
                    result.append(frame)
                    if len(result) == limit:
                        break
            return list(reversed(result))
        return [
            frame
            for frame in sequence[:position]
            if frame.exposure_end_us <= current.exposure_start_us
        ]

    def resolve(
        self,
        sample_token: str,
        *,
        control: TemporalControl = "correct_past",
        seed: int = 7,
        delay_frames: int = 1,
    ) -> TemporalRGBSample:
        """Resolve two history slots without future or cross-sequence support."""
        if control not in CONTROLS:
            raise ValueError(f"Unknown temporal control: {control}")
        if sample_token not in self._tokens:
            raise KeyError(f"Unknown sample token: {sample_token}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(delay_frames, bool) or not isinstance(delay_frames, int) or delay_frames < 1:
            raise ValueError("delay_frames must be a positive integer")
        current = self._tokens[sample_token]
        chosen: list[RGBExposure]
        if control == "current_only":
            chosen = []
        elif control == "current_repeated":
            chosen = [current, current]
        elif control == "correct_past":
            chosen = self._causal_past(current, limit=2)
        elif control == "delayed_past":
            causal = self._causal_past(current, limit=2 + delay_frames)
            chosen = causal[: max(0, len(causal) - delay_frames)][-2:]
        else:
            causal = self._causal_past(current)
            chosen = sorted(
                causal,
                key=lambda frame: (
                    hashlib.sha256(
                        f"{seed}:{sample_token}:{frame.sample_token}".encode()
                    ).hexdigest(),
                    frame.sample_token,
                ),
            )[:2]
        padded: list[RGBExposure | None] = [None] * (2 - len(chosen)) + chosen
        tokens = tuple(frame.sample_token if frame else None for frame in padded)
        ends = tuple(frame.exposure_end_us if frame else None for frame in padded)
        ages = tuple(
            current.exposure_end_us - frame.exposure_end_us if frame else None for frame in padded
        )
        mask = tuple(frame is not None for frame in padded)
        result = TemporalRGBSample(
            sequence_id=current.sequence_id,
            current_token=current.sample_token,
            current_exposure_start_us=current.exposure_start_us,
            current_exposure_end_us=current.exposure_end_us,
            decision_sensor_us=current.exposure_end_us,
            history_tokens=tokens,  # type: ignore[arg-type]
            history_mask=mask,  # type: ignore[arg-type]
            history_exposure_end_us=ends,  # type: ignore[arg-type]
            history_age_us=ages,  # type: ignore[arg-type]
            control=control,
        )
        self.validate_sample(result)
        return result

    def validate_sample(self, sample: TemporalRGBSample) -> None:
        """Reject future, unknown or cross-sequence history references."""
        current = self._tokens.get(sample.current_token)
        if current is None or current.sequence_id != sample.sequence_id:
            raise ValueError("Temporal sample current identity differs from the index")
        if (
            sample.current_exposure_start_us != current.exposure_start_us
            or sample.current_exposure_end_us != current.exposure_end_us
            or sample.decision_sensor_us != current.exposure_end_us
        ):
            raise ValueError("Temporal sample current timestamps differ from the index")
        for index, token in enumerate(sample.history_tokens):
            if token is None:
                continue
            frame = self._tokens.get(token)
            if frame is None or frame.sequence_id != current.sequence_id:
                raise ValueError("Temporal history is unknown or crosses a sequence")
            expected_age = current.exposure_end_us - frame.exposure_end_us
            if (
                sample.history_exposure_end_us[index] != frame.exposure_end_us
                or sample.history_age_us[index] != expected_age
            ):
                raise ValueError("Temporal history timestamps differ from the index")
            if token != current.sample_token and frame.exposure_end_us > current.exposure_start_us:
                raise ValueError("Temporal history is not complete before the current exposure")
            if token == current.sample_token and sample.control != "current_repeated":
                raise ValueError("Only the repeated-current control may reuse the current frame")


def audit_temporal_support(
    index: TemporalRGBIndex, assignments: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Count causal history support for frozen sequence roles."""
    owners: dict[str, str] = {}
    for role, sequences in assignments.items():
        for sequence in sequences:
            if sequence in owners:
                raise ValueError(f"Sequence crosses temporal roles: {sequence}")
            owners[sequence] = role
    if set(owners) != set(index.sequence_ids):
        raise ValueError("Temporal roles must cover exactly the indexed sequences")
    counts: dict[str, Counter[str]] = {role: Counter() for role in assignments}
    ages: dict[str, list[int]] = {role: [] for role in assignments}
    for token in index.sample_tokens:
        sequence = index._tokens[token].sequence_id
        role = owners[sequence]
        sample = index.resolve(token, control="correct_past")
        support = sum(sample.history_mask)
        counts[role][f"history_{support}"] += 1
        counts[role]["frames"] += 1
        ages[role].extend(age for age in sample.history_age_us if age is not None)
    return {
        "schema": "eap_detection_temporal_support_v1",
        "availability_scope": "sensor exposure timestamps only; arrival unavailable",
        "history_rule": "previous two same-sequence frames with exposure_end <= current_start",
        "roles": {
            role: {
                **dict(counter),
                "sequence_count": len(assignments[role]),
                "history_age_us_min": min(ages[role]) if ages[role] else None,
                "history_age_us_max": max(ages[role]) if ages[role] else None,
            }
            for role, counter in counts.items()
        },
    }
