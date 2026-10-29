"""A1 optimizer state and captured model retained across bounded accounting scopes."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.common import digest, freeze, read, sha, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from . import adaptation_depth as a1
from .common import accounting, require_resources
from .fast_runtime import PersistentSensors
from .geometry_capture import contiguous_subset
from .phase_transport_loss import backward
from .train import make_optimizer, setup


class Session:
    """One arm/model/optimizer per process; scopes do not rebuild its CUDA graphs."""

    def __init__(
        self, directory: Path, mode: str, masses: np.ndarray, *, synthetic: bool = False
    ) -> None:
        a1.registered()
        setup(7)
        model, parent_sha = a1.make_model(mode, 2, synthetic=synthetic)
        self.model = model.to("cuda").train()
        self.directory, self.mode, self.masses = directory, mode, masses
        self.endpoint = 10 if synthetic else a1.UPDATES
        identity = {
            "fit_id": directory.name,
            "mode": mode,
            "stage": 2,
            "registration_sha256": sha(a1.REGISTRATION),
            "endpoint": self.endpoint,
            "population": "synthetic8" if synthetic else "P",
            "device": "cuda",
            "parent_sha256": parent_sha,
            "mass_sha256": digest(masses.tolist()),
        }
        identity["identity_sha256"] = digest(identity)
        freeze(directory / "IDENTITY.json", identity)
        self.optimizer, self.scheduler = make_optimizer(self.model, updates=a1.UPDATES)
        self.sampler = torch.Generator().manual_seed(7)
        self.state = ProducerCheckpoint(directory, identity, self.endpoint)
        cursor = (
            self.state.restore(self.model, self.optimizer, self.scheduler, self.sampler)
            if self.state.pointer.exists()
            else None
        )
        self.curve, self.timings = (cursor["curve"], cursor["timings"]) if cursor else ([], [])
        self.profile: list[dict] = []
        self.scope_count, self.capture_count = 0, 0
        if cursor is None:
            self.save("RUNNING")

    def save(self, status: str) -> None:
        """Keep the original atomic checkpoint format, cursor and complete RNG state."""
        self.state.save(
            self.model,
            self.optimizer,
            self.scheduler,
            self.sampler,
            {"curve": self.curve, "timings": self.timings},
            status=status,
        )

    def advance(self, sensors: PersistentSensors, *, limit: int, deadline: float) -> dict:
        """Advance the unchanged sampled objective with an already-live model and cache."""
        accounting()
        self.scope_count += 1
        sensors.prepare(self.sampler.get_state(), pinned=True)
        try:
            while (
                self.state.completed < min(limit, self.endpoint) and time.perf_counter() < deadline
            ):
                if self.state.completed % 20 == 0:
                    try:
                        require_resources()
                    except RuntimeError as error:
                        if str(error).startswith("RESOURCE_WAIT"):
                            break
                        raise
                tick = time.perf_counter()
                ids = torch.randint(sensors.population, (32,), generator=self.sampler).tolist()
                batch = sensors.batch(ids)
                ready = time.perf_counter()
                batches = [
                    contiguous_subset(batch, list(range(i, i + 4))).to(torch.device("cuda"))
                    for i in range(0, 32, 4)
                ]
                transferred = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                was_captured = getattr(self.model, "_v18_capture_active", False)
                losses = backward(
                    self.model,
                    batches,
                    torch.from_numpy(self.masses[ids]).float(),
                    sensors.population,
                )
                self.capture_count += int(not was_captured)
                differentiated = time.perf_counter()
                if self.state.completed == 0:
                    gradients = {
                        name: float(p.grad.norm()) if p.grad is not None else 0.0
                        for name, p in self.model.encoder.named_parameters()
                    }
                    if not all(np.isfinite(list(gradients.values()))) or not any(
                        gradients.values()
                    ):
                        raise ValueError("A1 encoder must receive finite nonzero gradients")
                    write(self.directory / "ENCODER_GRADIENT.json", gradients)
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0, error_if_nonfinite=True
                )
                self.state.begin()
                self.optimizer.step()
                self.scheduler.step()
                self.state.commit()
                self.curve.append(
                    {"update": self.state.completed, "ids": ids, **losses, "norm": float(norm)}
                )
                elapsed = time.perf_counter() - tick
                self.timings.append(elapsed)
                self.profile.append(
                    {
                        "update": self.state.completed,
                        "input_wait_s": ready - tick,
                        "transfer_submit_s": transferred - ready,
                        "forward_backward_sync_s": differentiated - transferred,
                        "optimizer_journal_s": elapsed - (differentiated - tick),
                        "total_s": elapsed,
                    }
                )
                if self.state.completed % 100 == 0:
                    self.save("RUNNING")
                    accounting()
                    self.publish_runtime(sensors)
                    print(
                        f"{self.directory.name} {self.state.completed}/{self.endpoint} "
                        f"loss={losses['loss']:.6g}",
                        flush=True,
                    )
        finally:
            sensors.close()
        self.save("COMPLETE" if self.state.completed == self.endpoint else "PAUSED")
        if self.curve:
            with (self.directory / "CURVE.csv").open("w", newline="") as stream:
                rows = [
                    {**{k: v for k, v in row.items() if k != "ids"}, "seconds": seconds}
                    for row, seconds in zip(self.curve, self.timings, strict=True)
                ]
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.publish_runtime(sensors)
        accounting()
        return read(self.directory / "CHECKPOINT_RECEIPT.json")

    def publish_runtime(self, sensors: PersistentSensors) -> None:
        """Publish bounded timings, scope totals and the original scientific curves."""
        write(
            self.directory / "PERSISTENT_RUNTIME.json",
            {
                "completed_updates": self.state.completed,
                "scopes_this_process": self.scope_count,
                "captures_this_process": self.capture_count,
                "model_parameter_ids": [id(p) for p in self.model.parameters()],
                "cache": sensors.counters() if hasattr(sensors, "rows_cached") else {},
                "recent_updates": self.profile[-100:],
                "timing_scope": (
                    "Host intervals; transfers may overlap. "
                    "No CUDA event synchronizations inserted."
                ),
            },
        )
        self.profile = self.profile[-100:]
