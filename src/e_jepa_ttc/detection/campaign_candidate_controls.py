"""Isolated, preregistrable controls for optional RGB-R3/R4 candidates.

This module is deliberately not imported by the frozen R0--R2 producer.  It
contains only bounded candidate mechanics that can be integrated after an
explicit recipe choice and a new source snapshot.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import torch


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PhotometricPolicy:
    """Moderate RGB-only jitter magnitudes fixed before candidate execution."""

    brightness: float
    contrast: float
    saturation: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not 0 <= float(value) <= 0.3:
                raise ValueError(f"{name} must be in the preregisterable range [0, 0.3]")

    @property
    def sha256(self) -> str:
        """Return the canonical policy hash used in candidate receipts."""
        return _canonical_sha256(asdict(self))


class StatefulPhotometricAugmenter:
    """Apply normalized RGB jitter using an isolated, checkpointable CPU RNG."""

    def __init__(self, policy: PhotometricPolicy, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Photometric seed must be a nonnegative integer")
        self.policy = policy
        self.seed = seed
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def apply(self, images: torch.Tensor, *, training: bool) -> torch.Tensor:
        """Transform images only; evaluation is an exact no-op."""
        if not training or not any(asdict(self.policy).values()):
            return images
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("Photometric input must be floating [B,3,H,W]")
        mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        rgb = (images * std + mean).clamp(0, 1)

        def factors(magnitude: float) -> torch.Tensor:
            values = (
                1 + (torch.rand((len(rgb), 1, 1, 1), generator=self.generator) * 2 - 1) * magnitude
            )
            return values.to(device=images.device, dtype=images.dtype)

        if self.policy.brightness:
            rgb = rgb * factors(self.policy.brightness)
        if self.policy.contrast:
            gray = rgb.mean(dim=(1, 2, 3), keepdim=True)
            rgb = gray + factors(self.policy.contrast) * (rgb - gray)
        if self.policy.saturation:
            gray = rgb.mean(dim=1, keepdim=True)
            rgb = gray + factors(self.policy.saturation) * (rgb - gray)
        return (rgb.clamp(0, 1) - mean) / std

    def state_dict(self) -> dict[str, Any]:
        """Serialize the exact next augmentation draw."""
        return {
            "schema": "eap_detection_photometric_rng_v1",
            "policy": asdict(self.policy),
            "policy_sha256": self.policy.sha256,
            "seed": self.seed,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore only a state belonging to the same preregistered policy."""
        if (
            state.get("schema") != "eap_detection_photometric_rng_v1"
            or state.get("policy_sha256") != self.policy.sha256
            or state.get("policy") != asdict(self.policy)
            or state.get("seed") != self.seed
        ):
            raise ValueError("Photometric RNG state belongs to another recipe")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("Photometric RNG state is malformed")
        self.generator.set_state(generator_state)


@dataclass(frozen=True)
class RareIdentityPolicy:
    """Fixed mixture and hard exposure caps for an optional R4 sampler."""

    rare_classes: tuple[str, ...]
    rare_draw_fraction: float
    max_frame_exposures_per_epoch: int = 4
    max_identity_exposure_multiplier: int = 4

    def __post_init__(self) -> None:
        if not self.rare_classes or len(set(self.rare_classes)) != len(self.rare_classes):
            raise ValueError("rare_classes must be nonempty and unique")
        if tuple(sorted(self.rare_classes)) != self.rare_classes:
            raise ValueError("rare_classes must be sorted for a canonical recipe")
        if not 0 <= self.rare_draw_fraction < 1:
            raise ValueError("rare_draw_fraction must be in [0, 1)")
        if not 1 <= self.max_frame_exposures_per_epoch <= 4:
            raise ValueError("Frame exposure cap must be between one and four")
        if not 1 <= self.max_identity_exposure_multiplier <= 4:
            raise ValueError("Identity exposure multiplier must be between one and four")

    @property
    def sha256(self) -> str:
        """Return the canonical sampler-policy hash."""
        return _canonical_sha256(asdict(self))


class CappedRareIdentitySampler:
    """Mix uniform train draws with identity-balanced, hard-capped rare draws.

    ``memberships`` maps a train token to ``(class_name, identity_id)`` pairs.
    Identity IDs must already include their sequence namespace.  Each epoch has
    exactly ``len(tokens)`` draws, preserving the common exposure/update budget.
    """

    def __init__(
        self,
        tokens: Sequence[str],
        memberships: Mapping[str, Sequence[tuple[str, str]]],
        *,
        policy: RareIdentityPolicy,
        seed: int,
        source_role: str,
        manifest_sha256: str,
    ) -> None:
        self.tokens = tuple(tokens)
        if not self.tokens or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Sampler tokens must be nonempty and unique")
        if source_role != "train":
            raise ValueError("Rare-identity membership may be derived only from train")
        if len(manifest_sha256) != 64:
            raise ValueError("Sampler requires the full train manifest SHA-256")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Sampler seed must be a nonnegative integer")
        unknown = set(memberships) - set(self.tokens)
        if unknown:
            raise ValueError(f"Membership contains tokens outside train: {sorted(unknown)[:3]}")
        self.policy = policy
        self.seed = seed
        self.source_role = source_role
        self.manifest_sha256 = manifest_sha256
        rare = set(policy.rare_classes)
        self.memberships = {
            token: tuple(
                sorted(
                    {
                        (class_name, identity)
                        for class_name, identity in memberships.get(token, ())
                        if class_name in rare
                    }
                )
            )
            for token in self.tokens
        }
        self.membership_sha256 = _canonical_sha256(
            {token: [list(item) for item in self.memberships[token]] for token in self.tokens}
        )
        if any(not identity for values in self.memberships.values() for _, identity in values):
            raise ValueError("Rare identity IDs must be nonempty")
        self.identity_frames: dict[tuple[str, str], tuple[str, ...]] = {}
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for token, values in self.memberships.items():
            for identity in values:
                grouped[identity].append(token)
        self.identity_frames = {key: tuple(sorted(value)) for key, value in grouped.items()}
        if not self.identity_frames:
            raise ValueError("R4 sampler requires at least one train rare identity")
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch = 0
        self.position = 0
        self.completed_epoch_receipts: list[dict[str, Any]] = []
        self.order, self.receipt = self._make_epoch()

    def _shuffle(self, values: Sequence[Any]) -> list[Any]:
        if not values:
            return []
        order = torch.randperm(len(values), generator=self.generator).tolist()
        return [values[index] for index in order]

    def _make_epoch(self) -> tuple[list[str], dict[str, Any]]:
        count = len(self.tokens)
        requested_rare = int(count * self.policy.rare_draw_fraction)
        uniform_order = self._shuffle(self.tokens)
        draws = list(uniform_order[: count - requested_rare])
        frame_exposures = Counter(draws)
        identity_exposures: Counter[tuple[str, str]] = Counter()
        for token in draws:
            identity_exposures.update(self.memberships[token])
        identity_caps = {
            identity: self.policy.max_identity_exposure_multiplier * len(frames)
            for identity, frames in self.identity_frames.items()
        }
        identities = self._shuffle(tuple(sorted(self.identity_frames)))
        candidate_frames = {
            identity: self._shuffle(self.identity_frames[identity]) for identity in identities
        }
        cursors = Counter()
        rare_draws = 0
        while rare_draws < requested_rare:
            made_progress = False
            for identity in identities:
                frames = candidate_frames[identity]
                for offset in range(len(frames)):
                    index = (cursors[identity] + offset) % len(frames)
                    token = frames[index]
                    affected = self.memberships[token]
                    if frame_exposures[token] >= self.policy.max_frame_exposures_per_epoch or any(
                        identity_exposures[item] >= identity_caps[item] for item in affected
                    ):
                        continue
                    draws.append(token)
                    frame_exposures[token] += 1
                    identity_exposures.update(affected)
                    cursors[identity] = index + 1
                    rare_draws += 1
                    made_progress = True
                    break
                if rare_draws == requested_rare:
                    break
            if not made_progress:
                break

        fallback = 0
        if len(draws) < count:
            for token in uniform_order[count - requested_rare :] + uniform_order:
                if len(draws) == count:
                    break
                affected = self.memberships[token]
                if frame_exposures[token] >= self.policy.max_frame_exposures_per_epoch or any(
                    identity_exposures[item] >= identity_caps[item] for item in affected
                ):
                    continue
                draws.append(token)
                frame_exposures[token] += 1
                identity_exposures.update(affected)
                fallback += 1
        if len(draws) != count:
            raise AssertionError("Candidate sampler did not preserve the exposure budget")
        draws = self._shuffle(draws)
        receipt = {
            "schema": "eap_detection_rare_identity_epoch_v1",
            "epoch": self.epoch,
            "source_role": self.source_role,
            "manifest_sha256": self.manifest_sha256,
            "membership_sha256": self.membership_sha256,
            "seed": self.seed,
            "policy": asdict(self.policy),
            "policy_sha256": self.policy.sha256,
            "total_exposures": count,
            "requested_rare_draws": requested_rare,
            "realized_rare_draws": rare_draws,
            "fallback_uniform_draws": fallback,
            "unique_frame_count": len(frame_exposures),
            "frame_exposures": dict(sorted(frame_exposures.items())),
            "identity_support_frames": {
                f"{category}|{identity}": len(self.identity_frames[(category, identity)])
                for category, identity in sorted(self.identity_frames)
            },
            "identity_exposures": {
                f"{category}|{identity}": identity_exposures[(category, identity)]
                for category, identity in sorted(self.identity_frames)
            },
            "order_sha256": _canonical_sha256(draws),
        }
        return draws, receipt

    def _validate_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        expected_epoch: int,
        order: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Recompute exposure ledgers and reject altered checkpoint evidence."""
        fields = {
            "schema",
            "epoch",
            "source_role",
            "manifest_sha256",
            "membership_sha256",
            "seed",
            "policy",
            "policy_sha256",
            "total_exposures",
            "requested_rare_draws",
            "realized_rare_draws",
            "fallback_uniform_draws",
            "unique_frame_count",
            "frame_exposures",
            "identity_support_frames",
            "identity_exposures",
            "order_sha256",
        }
        if set(receipt) != fields:
            raise ValueError("Rare sampler receipt fields differ from the contract")
        fixed = {
            "schema": "eap_detection_rare_identity_epoch_v1",
            "epoch": expected_epoch,
            "source_role": self.source_role,
            "manifest_sha256": self.manifest_sha256,
            "membership_sha256": self.membership_sha256,
            "seed": self.seed,
            "policy": asdict(self.policy),
            "policy_sha256": self.policy.sha256,
            "total_exposures": len(self.tokens),
            "requested_rare_draws": int(len(self.tokens) * self.policy.rare_draw_fraction),
        }
        if any(receipt.get(name) != value for name, value in fixed.items()):
            raise ValueError("Rare sampler receipt metadata differs from its population")
        raw_frames = receipt.get("frame_exposures")
        if not isinstance(raw_frames, Mapping):
            raise ValueError("Rare sampler frame exposure ledger is malformed")
        frame_exposures: dict[str, int] = {}
        for token, value in raw_frames.items():
            if (
                not isinstance(token, str)
                or token not in self.memberships
                or isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= self.policy.max_frame_exposures_per_epoch
            ):
                raise ValueError("Rare sampler frame exposure ledger is malformed")
            frame_exposures[token] = value
        if sum(frame_exposures.values()) != len(self.tokens) or receipt.get(
            "unique_frame_count"
        ) != len(frame_exposures):
            raise ValueError("Rare sampler frame exposures do not preserve the epoch budget")
        if order is not None:
            if Counter(order) != Counter(frame_exposures):
                raise ValueError("Rare sampler order differs from its frame exposure ledger")
            if receipt.get("order_sha256") != _canonical_sha256(list(order)):
                raise ValueError("Rare sampler checkpoint order differs from its receipt")
        elif not isinstance(receipt.get("order_sha256"), str) or len(receipt["order_sha256"]) != 64:
            raise ValueError("Completed rare sampler order hash is malformed")

        expected_support = {
            f"{category}|{identity}": len(self.identity_frames[(category, identity)])
            for category, identity in sorted(self.identity_frames)
        }
        identity_counts: Counter[tuple[str, str]] = Counter()
        for token, exposure in frame_exposures.items():
            for identity in self.memberships[token]:
                identity_counts[identity] += exposure
        expected_exposures = {
            f"{category}|{identity}": identity_counts[(category, identity)]
            for category, identity in sorted(self.identity_frames)
        }
        if (
            receipt.get("identity_support_frames") != expected_support
            or receipt.get("identity_exposures") != expected_exposures
        ):
            raise ValueError("Rare sampler identity exposure ledger is inconsistent")
        if any(
            expected_exposures[name] > self.policy.max_identity_exposure_multiplier * support
            for name, support in expected_support.items()
        ):
            raise ValueError("Rare sampler identity exposure cap was exceeded")
        realized = receipt.get("realized_rare_draws")
        fallback = receipt.get("fallback_uniform_draws")
        requested = fixed["requested_rare_draws"]
        if (
            isinstance(realized, bool)
            or not isinstance(realized, int)
            or not 0 <= realized <= requested
            or isinstance(fallback, bool)
            or not isinstance(fallback, int)
            or fallback != requested - realized
        ):
            raise ValueError("Rare sampler rare/fallback draw accounting is inconsistent")
        return dict(receipt)

    def next_batch(self, batch_size: int) -> list[str]:
        """Return the next exact token batch, crossing fixed-size epochs."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        result: list[str] = []
        while len(result) < batch_size:
            available = min(batch_size - len(result), len(self.order) - self.position)
            result.extend(self.order[self.position : self.position + available])
            self.position += available
            if self.position == len(self.order):
                self.completed_epoch_receipts.append(deepcopy(self.receipt))
                self.epoch += 1
                self.position = 0
                self.order, self.receipt = self._make_epoch()
        return result

    def state_dict(self) -> dict[str, Any]:
        """Serialize exact continuation and the effective exposure ledger."""
        return {
            "schema": "eap_detection_rare_identity_sampler_v1",
            "seed": self.seed,
            "manifest_sha256": self.manifest_sha256,
            "membership_sha256": self.membership_sha256,
            "policy_sha256": self.policy.sha256,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order,
            "receipt": deepcopy(self.receipt),
            "completed_epoch_receipts": deepcopy(self.completed_epoch_receipts),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore only an exact state from this population and recipe."""
        if (
            state.get("schema") != "eap_detection_rare_identity_sampler_v1"
            or state.get("seed") != self.seed
            or state.get("manifest_sha256") != self.manifest_sha256
            or state.get("membership_sha256") != self.membership_sha256
            or state.get("policy_sha256") != self.policy.sha256
        ):
            raise ValueError("Rare sampler state belongs to another preregistration")
        order = list(state.get("order", ()))
        if len(order) != len(self.tokens) or any(token not in self.memberships for token in order):
            raise ValueError("Rare sampler checkpoint order is malformed")
        position = state.get("position")
        epoch = state.get("epoch")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < len(order)
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise ValueError("Rare sampler checkpoint cursor is malformed")
        generator_state = state.get("generator_state")
        receipt = state.get("receipt")
        completed = state.get("completed_epoch_receipts")
        if (
            not isinstance(generator_state, torch.Tensor)
            or not isinstance(receipt, Mapping)
            or not isinstance(completed, Sequence)
            or isinstance(completed, (str, bytes))
        ):
            raise ValueError("Rare sampler checkpoint state is incomplete")
        if len(completed) != epoch:
            raise ValueError("Rare sampler completed epoch ledger is incomplete")
        validated_completed = []
        for expected_epoch, item in enumerate(completed):
            if not isinstance(item, Mapping):
                raise ValueError("Rare sampler completed epoch receipt is malformed")
            validated_completed.append(self._validate_receipt(item, expected_epoch=expected_epoch))
        validated_receipt = self._validate_receipt(receipt, expected_epoch=epoch, order=order)
        self.epoch = epoch
        self.position = position
        self.order = order
        self.receipt = validated_receipt
        self.completed_epoch_receipts = validated_completed
        self.generator.set_state(generator_state)
