"""Exactly 80 synthetic optimizer updates for prospective engine/resume checks.

No OLD_DEV inputs or targets are constructed or read. Run in a separate process
only after the previous heavy trainer exits. Interrupted technical reservations
remain conservative and require diagnosis instead of an automatic duplicate run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .engine import (
    N1,
    OUT,
    Lease,
    Resources,
    _owner,
    atomic_json,
    digest,
    fit_arm,
    protocol,
    publish_json,
    record,
    validate_pins,
)

if TYPE_CHECKING:
    from torch import Tensor


class SyntheticSource:
    """Deterministic synthetic normalized features; masses belong to full population."""

    def __init__(self) -> None:
        import torch

        from e_jepa_ttc.simplex_t.training import state_digest

        generator = torch.Generator().manual_seed(991)
        self.features = torch.randn(37, 8, 17, generator=generator)
        self.times = torch.rand(37, 8, 4, generator=generator)
        self.valid = torch.ones(37, 8, dtype=torch.bool)
        self.experts = 0.01 + 0.08 * torch.rand(37, 3, generator=generator)
        self.truth = -0.02 + 0.15 * torch.rand(37, generator=generator)
        weights = torch.arange(1, 38, dtype=torch.float32)
        self.mass = weights / weights.sum()
        self.identity_sha256 = state_digest(
            dict(
                synthetic_only=True,
                generator_seed=991,
                features=self.features,
                times=self.times,
                valid=self.valid,
                experts=self.experts,
                truth=self.truth,
                mass=self.mass,
            )
        )

    @property
    def population(self) -> int:
        return 37

    def gather(self, query_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return tuple(
            value[query_ids]
            for value in (
                self.features,
                self.times,
                self.valid,
                self.experts,
                self.truth,
                self.mass,
            )
        )  # type: ignore[return-value]


def segments() -> list[dict]:
    """Six fixed optimizer-call segments; never a seventh seed or scientific fit."""
    return [
        dict(
            key=f"{arm}/{mode}",
            arm=arm,
            mode=mode,
            start=start,
            stop=stop,
            authorized_updates=stop - start,
        )
        for arm in ("FULL_C0", "SET_AGE_C0")
        for mode, start, stop in (("continuous", 0, 20), ("partial", 0, 10), ("resume", 10, 20))
    ]


class TechnicalJournal:
    """Durable reservations and confirmed checkpoint work, separate from science."""

    def __init__(self, segment: dict) -> None:
        self.segment = segment
        self.path = OUT / "TECHNICAL_WORK.json"

    def start(self, completed: int, checkpoint: Path) -> None:
        if (
            not self.segment["start"] <= completed <= self.segment["stop"]
            or not checkpoint.is_file()
        ):
            raise ValueError("technical segment cannot rewind or duplicate work")
        state = record(self.path)
        slot = state["segments"][self.segment["key"]]
        if slot["status"] not in {"REGISTERED_NOT_EXECUTED", "PAUSED"}:
            raise ValueError("technical work already attempted: diagnose before retry")
        if completed != self.segment["start"] + slot["confirmed_updates"]:
            raise ValueError("technical checkpoint and confirmed work differ")
        slot.update(
            status="ACTIVE_RESERVED",
            checkpoint=str(checkpoint),
            confirmed_updates=slot["confirmed_updates"],
            possible_unconfirmed_upper=self.segment["stop"] - completed,
        )
        atomic_json(self.path, state)

    def before_update(self, completed: int) -> None:
        if not self.segment["start"] <= completed < self.segment["stop"]:
            raise ValueError("technical update outside its fixed reservation")

    def checkpoint_saved(self, completed: int, checkpoint: Path) -> None:
        if not self.segment["start"] <= completed <= self.segment["stop"]:
            raise ValueError("technical checkpoint outside its reservation")
        state = record(self.path)
        slot = state["segments"][self.segment["key"]]
        slot.update(
            confirmed_updates=completed - self.segment["start"],
            checkpoint_sha256=digest(checkpoint),
            possible_unconfirmed_upper=self.segment["stop"] - completed,
            status="CONFIRMED" if completed == self.segment["stop"] else "PAUSED",
        )
        state["confirmed_updates"] = sum(v["confirmed_updates"] for v in state["segments"].values())
        state["physical_lower"] = state["confirmed_updates"]
        state["physical_upper"] = state["reserved_updates"]
        atomic_json(self.path, state)


def main() -> int:
    """Test complete-state equivalence without ever invoking scientific endpoints."""
    from e_jepa_ttc.simplex_t.training import load_checkpoint, state_digest

    from .model import MaskedSource

    resources = Resources(training=True)
    resources.check()
    if _owner(N1 / "WRITER.lock") is not None:
        raise RuntimeError("N1 heavy owner is alive; defer synthetic optimizer probes")
    p, pin = protocol()
    validate_pins(p, full=True)
    receipt = OUT / "verification/ENGINE_RESUME.json"
    if receipt.exists():
        proof = record(receipt)
        if (
            proof["status"] != "EXACT_SYNTHETIC_RESUME_EQUIVALENCE"
            or proof["optimizer_updates"] != 80
        ):
            raise ValueError("conflicting technical completion receipt")
        print("TECHNICAL_PROOF_ALREADY_COMPLETE_NO_UPDATES", flush=True)
        return 0
    path = OUT / "TECHNICAL_WORK.json"
    ledger: dict = dict(
        schema="cost_context_technical_work_v1",
        protocol_sha256=pin,
        synthetic_only=True,
        reserved_updates=80,
        cap=200,
        confirmed_updates=0,
        physical_lower=0,
        physical_upper=80,
        scientific_updates=0,
        segments={
            row["key"]: dict(
                **row,
                status="REGISTERED_NOT_EXECUTED",
                confirmed_updates=0,
                possible_unconfirmed_upper=0,
            )
            for row in segments()
        },
    )
    with Lease():
        if path.exists():
            old = record(path)
            if (
                old["protocol_sha256"] != pin
                or old["reserved_updates"] != 80
                or set(old["segments"]) != set(ledger["segments"])
            ):
                raise ValueError("technical authority changed")
            if any(v["status"] == "ACTIVE_RESERVED" for v in old["segments"].values()):
                raise RuntimeError(
                    "uncertain technical crash work requires diagnosis; no automatic replay"
                )
        else:
            publish_json(path, ledger)
        evidence = []
        for row in segments():
            source = MaskedSource(SyntheticSource(), row["arm"])
            folder = (
                OUT
                / "verification/technical"
                / row["arm"]
                / ("continuous" if row["mode"] == "continuous" else "interrupted")
            )
            slot = record(path)["segments"][row["key"]]
            if slot["status"] != "CONFIRMED":
                result = fit_arm(
                    source,
                    row["arm"],
                    folder,
                    seed=7,
                    freeze_sha256=pin,
                    resource_ok=resources,
                    resume=row["mode"] == "resume" or slot["status"] == "PAUSED",
                    stop_after=row["stop"],
                    journal=TechnicalJournal(row),
                )
                if result["completed_updates"] != row["stop"]:
                    return 3
            if row["mode"] == "resume":
                continuous = load_checkpoint(folder.parent / "continuous/checkpoint_last.pt")
                interrupted = load_checkpoint(folder / "checkpoint_last.pt")
                if state_digest(continuous) != state_digest(interrupted):
                    raise ValueError("continuous/resumed complete-state mismatch")
                evidence.append(
                    dict(
                        arm=row["arm"],
                        complete_state_sha256=state_digest(continuous),
                        continuous_checkpoint_sha256=digest(
                            folder.parent / "continuous/checkpoint_last.pt"
                        ),
                        interrupted_checkpoint_sha256=digest(folder / "checkpoint_last.pt"),
                        model_optimizer_rng_sampler_losses_equal=True,
                        continuous_updates=20,
                        interrupted_updates=20,
                    )
                )
        final = record(path)
        if final["confirmed_updates"] != 80:
            raise ValueError("technical work accounting differs from 80 authorized updates")
        publish_json(
            receipt,
            dict(
                status="EXACT_SYNTHETIC_RESUME_EQUIVALENCE",
                protocol_sha256=pin,
                optimizer_updates=80,
                scientific_updates=0,
                synthetic_only=True,
                families=evidence,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
