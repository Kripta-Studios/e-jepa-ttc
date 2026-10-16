"""Optional exact-loss-graph overlay on the admitted persistent A1 coordinator."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha, write

from . import adaptation_runtime as base
from . import persistent_adaptation as persistent
from .captured_phase_loss import backward
from .captured_phase_loss_qa import OUT, QA, checked_registration
from .fast_runtime import PersistentSensors

MODULE = "operational.joint_measurement.adaptation_loss_graph_runtime"


def admission() -> dict:
    """Require every exactness/speed proof; never infer admission from an existing file."""
    record = checked_registration()
    proofs = {"native": OUT / "NATIVE.json"}
    proofs.update({mode: QA / mode / "RESULT.json" for mode in base.a1.MODES})
    for path in proofs.values():
        proof = read(path)
        if proof["status"] != "PASS" or proof["registration_sha256"] != sha(
            OUT / "REGISTRATION.json"
        ):
            raise ValueError(f"Loss graph proof not admitted: {path}")
    result = {
        "status": "ADMITTED",
        "sources": record["sources"],
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "proofs": {key: sha(path) for key, path in proofs.items()},
        "prior_runtime": base.admission(),
        "scientific_recipe_changed": False,
    }
    freeze(OUT / "ADMISSION.json", result)
    return result


class LossGraphSession(persistent.Session):
    """Keep the checkpoint schema while recording the additional ephemeral graph."""

    def __init__(
        self, directory: Path, mode: str, masses: np.ndarray, *, synthetic: bool = False
    ) -> None:
        if synthetic:
            raise ValueError("Production overlay requires separate synthetic QA, not this session")
        self.transition_origin: dict | None = None
        super().__init__(directory, mode, masses, synthetic=synthetic)
        freeze(directory / "LOSS_GRAPH_ADMISSION.json", admission())
        if mode == "LAST-BLOCK" and not (OUT / "ACTIVATION_VERIFIED.json").exists():
            origin = torch.load(OUT / "ORIGIN.pt", map_location="cpu", weights_only=False)
            if self.state.completed != origin["completed_updates"]:
                raise ValueError("Unverified scientific transition cursor changed")
            self.transition_origin = origin

    def save(self, status: str) -> None:
        """Verify complete-state continuity at the first durable scientific checkpoint."""
        super().save(status)
        if self.transition_origin is not None:
            with patch.object(base, "OUT", OUT):
                base.verify_transition(self.directory, self.transition_origin, len(self.masses))
            if (OUT / "ACTIVATION_VERIFIED.json").exists():
                self.transition_origin = None

    def publish_runtime(self, sensors: PersistentSensors) -> None:
        """Report both captures without changing the previous runtime's evidence schema."""
        super().publish_runtime(sensors)
        loss = getattr(self.model, "_v18_phase_loss", None)
        write(
            self.directory / "LOSS_GRAPH_RUNTIME.json",
            {
                "completed_updates": self.state.completed,
                "model_captures_this_process": self.capture_count,
                "loss_captures_this_process": loss.capture_count if loss is not None else 0,
                "scopes_this_process": self.scope_count,
                "loss_graph_id": id(loss) if loss is not None else None,
                "admission_sha256": sha(OUT / "ADMISSION.json"),
            },
        )


def child(mode: str) -> None:
    """Use the original child loop, source admission, resource caps and state ownership."""
    admission()
    with (
        patch.object(base, "Session", LossGraphSession),
        patch.object(persistent, "backward", backward),
    ):
        base.child(mode)


def child_command(command: list[str]) -> list[str]:
    """Replace only the registered child entry point, preserving all other arguments."""
    expected = "operational.joint_measurement.adaptation_runtime"
    if len(command) != 8 or command[3:6] != ["-m", expected, "child"] or command[6] != "--arm":
        raise ValueError("Unexpected persistent child command")
    if command[7] not in base.a1.MODES:
        raise ValueError("Unknown A1 arm")
    return [*command[:4], MODULE, *command[5:]]


def run() -> None:
    """Reuse the sole coordinator lock and arm sequence; launch only the admitted overlay."""
    admission()

    def launch(command: list[str], *, cwd: Path, creationflags: int) -> subprocess.Popen:
        return subprocess.Popen(child_command(command), cwd=cwd, creationflags=creationflags)

    proxy = SimpleNamespace(Popen=launch, CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW)
    with patch.object(base, "subprocess", proxy):
        base.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("admit", "run", "child"))
    parser.add_argument("--arm", choices=base.a1.MODES, default=base.a1.MODES[0])
    args = parser.parse_args()
    child(args.arm) if args.action == "child" else {"admit": admission, "run": run}[args.action]()
