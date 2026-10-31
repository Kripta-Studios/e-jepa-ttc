"""Train-only, finite replay schedules with explicit frame and identity caps."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA = "phase_detection_capped_rare_schedule_v1"


def canonical_hash(value: object) -> str:
    """Hash a JSON value independently of formatting."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_schedule(
    tokens: list[str],
    rows: dict[str, dict[str, Any]],
    baseline: list[int],
    *,
    seed: int,
    multiplier: int = 4,
    categories: tuple[str, ...] = ("bicycle", "other_vehicle"),
) -> dict[str, Any]:
    """Replace common exposures with rare ones, preserving every unique train frame."""
    if not tokens or len(set(tokens)) != len(tokens) or set(rows) != set(tokens):
        raise ValueError("Expected unique train tokens and exactly their population rows")
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError("Exposure multiplier must be a positive integer")
    if not baseline or any(type(i) is not int or not 0 <= i < len(tokens) for i in baseline):
        raise ValueError("Invalid baseline indices")
    counts = Counter(baseline)
    if len(counts) != len(tokens) or len(set(counts.values())) != 1:
        raise ValueError("Baseline must contain complete uniform passes")
    identities: list[list[str]] = []
    for token in tokens:
        row = rows[token]
        sequence = token.split(":", 1)[0]
        if row["role"] != "train" or row["sequence_id"] != sequence or row["sample_token"] != token:
            raise ValueError("Sampling metadata must belong to the exact train token")
        if row["trainer_overflow"]:
            raise ValueError("Overflow frames are not admitted")
        identities.append(
            sorted(
                {
                    json.dumps([sequence, category, str(identity)], separators=(",", ":"))
                    for category in categories
                    for identity in row["trainable_identities"].get(category, [])
                }
            )
        )
    rare = [index for index, ids in enumerate(identities) if ids]
    if not rare:
        raise ValueError("No admitted rare identities in train")
    repeats = [i for i in rare for _ in range((multiplier - 1) * counts[i])]
    rng = random.Random(seed)
    rng.shuffle(repeats)
    donors = [position for position, index in enumerate(baseline) if not identities[index]]
    rng.shuffle(donors)
    schedule = list(baseline)
    after = counts.copy()
    donor_cursor = 0
    for rare_index in repeats:
        while donor_cursor < len(donors) and after[baseline[donors[donor_cursor]]] <= 1:
            donor_cursor += 1
        if donor_cursor == len(donors):
            raise ValueError(
                "Insufficient common exposures to achieve cap while retaining diversity"
            )
        position = donors[donor_cursor]
        donor_cursor += 1
        after[schedule[position]] -= 1
        schedule[position] = rare_index
        after[rare_index] += 1

    def identity_counts(indices: list[int]) -> dict[str, int]:
        return dict(
            sorted(Counter(identity for i in indices for identity in identities[i]).items())
        )

    before_ids, after_ids = identity_counts(baseline), identity_counts(schedule)
    if any(after_ids[key] > multiplier * value for key, value in before_ids.items()):
        raise AssertionError("Identity cap exceeded")
    return {
        "schema": SCHEMA,
        "seed": seed,
        "tokens": tokens,
        "indices": schedule,
        "baseline_indices_sha256": canonical_hash(baseline),
        "identities_by_index": identities,
        "baseline_frame_exposures": [counts[i] for i in range(len(tokens))],
        "frame_exposures": [after[i] for i in range(len(tokens))],
        "baseline_identity_exposures": before_ids,
        "identity_exposures": after_ids,
        "audit": {
            "target_classes": list(categories),
            "maximum_multiplier": multiplier,
            "exposures": len(schedule),
            "unique_frames": len(after),
            "rare_frames": len(rare),
            "rare_identities": len(before_ids),
            "replaced_common_exposures": len(repeats),
            "baseline_rare_frame_exposures": sum(counts[i] for i in rare),
            "rare_frame_exposures": sum(after[i] for i in rare),
            "train_labels_only": True,
        },
    }


class ReplayScheduleSampler:
    """Consume a hash-bound finite calendar with exact checkpoint continuation."""

    def __init__(self, indices: list[int], size: int, seed: int, plan_sha256: str) -> None:
        if not indices or size < 1 or any(type(i) is not int or not 0 <= i < size for i in indices):
            raise ValueError("Invalid replay calendar")
        self.indices = tuple(indices)
        self.size, self.seed, self.plan_sha256 = size, seed, plan_sha256
        self.cursor = 0

    @property
    def epoch(self) -> int:
        """Return effective passes, without implying uniform per-frame exposure."""
        return self.cursor // self.size

    def next_batch(self, batch_size: int) -> list[int]:
        """Consume one batch; never wrap or silently extend the registered budget."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        end = self.cursor + batch_size
        if end > len(self.indices):
            raise ValueError("Registered sampling budget exhausted")
        result = list(self.indices[self.cursor : end])
        self.cursor = end
        return result

    def state_dict(self) -> dict[str, Any]:
        """Serialize cursor and exact calendar identity."""
        return {
            "schema": SCHEMA,
            "size": self.size,
            "seed": self.seed,
            "plan_sha256": self.plan_sha256,
            "cursor": self.cursor,
            "indices_sha256": canonical_hash(self.indices),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Reject changed plans, budgets and corrupt cursors before continuation."""
        expected = self.state_dict()
        if set(state) != set(expected) or any(
            state[k] != v for k, v in expected.items() if k != "cursor"
        ):
            raise ValueError("Sampler state belongs to another registered schedule")
        cursor = state["cursor"]
        if type(cursor) is not int or not 0 <= cursor <= len(self.indices):
            raise ValueError("Invalid sampler cursor")
        self.cursor = cursor


def load_schedule(
    path: Path,
    *,
    expected_sha256: str,
    tokens: Sequence[str],
    seed: int,
    updates: int,
    batch_size: int,
    manifest_sha256: str,
) -> tuple[ReplayScheduleSampler, dict[str, Any]]:
    """Admit an immutable schedule and independently recount its actual exposure caps."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Sampling plan file hash differs")
    plan = json.loads(raw)
    if (
        plan["schema"] != SCHEMA
        or plan["tokens"] != list(tokens)
        or plan["seed"] != seed
        or plan["manifest_sha256"] != manifest_sha256
        or len(plan["indices"]) != updates * batch_size
    ):
        raise ValueError("Sampling plan does not match campaign population or budget")
    sampler = ReplayScheduleSampler(plan["indices"], len(tokens), seed, expected_sha256)
    counts = Counter(plan["indices"])
    actual = [counts[i] for i in range(len(tokens))]
    before = plan["baseline_frame_exposures"]
    multiplier = plan["audit"]["maximum_multiplier"]
    if (
        actual != plan["frame_exposures"]
        or len(before) != len(tokens)
        or any(type(n) is not int or n < 1 for n in before)
        or len(set(before)) != 1
        or sum(before) != updates * batch_size
        or type(multiplier) is not int
        or not 1 <= multiplier <= 4
        or any(not 1 <= n <= multiplier * b for n, b in zip(actual, before, strict=True))
    ):
        raise ValueError("Materialized frame exposures violate the registered cap")
    identities = plan["identities_by_index"]
    if len(identities) != len(tokens):
        raise ValueError("Identity mapping differs from population")
    before_ids: Counter[str] = Counter()
    after_ids: Counter[str] = Counter()
    for i, ids in enumerate(identities):
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated identity within a frame")
        for identity in ids:
            before_ids[identity] += before[i]
            after_ids[identity] += actual[i]
    if (
        dict(before_ids) != plan["baseline_identity_exposures"]
        or dict(after_ids) != plan["identity_exposures"]
        or any(after_ids[key] > multiplier * value for key, value in before_ids.items())
    ):
        raise ValueError("Materialized identity exposures violate the registered cap")
    return sampler, plan["audit"] | {"sampling_plan_sha256": expected_sha256}
