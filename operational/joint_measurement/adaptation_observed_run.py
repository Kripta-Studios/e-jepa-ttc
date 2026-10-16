"""Repair boundary weight evidence without changing the admitted training implementation."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from operational.range_transfer.common import freeze, read, sha, write

from . import adaptation_depth as a1
from . import adaptation_loss_graph_runtime as runtime
from .common import ROOT, RUN

OUT = RUN / "runtime/A1_boundary_evidence_v1"
MODULE = "operational.joint_measurement.adaptation_observed_run"
ORIGINAL_VERIFY = a1.verify_change


def differences(
    current: Mapping[str, torch.Tensor], parent: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    """Compare complete CPU state dictionaries, rejecting missing/nonfinite tensors."""
    if current.keys() != parent.keys():
        raise ValueError("Weight witness state keys differ")
    result = {}
    for name, value in current.items():
        before = parent[name]
        if value.shape != before.shape or value.dtype != before.dtype:
            raise ValueError(f"Weight witness tensor contract differs: {name}")
        delta = float((value - before).abs().max())
        if not torch.isfinite(torch.tensor(delta)):
            raise ValueError(f"Nonfinite weight difference: {name}")
        result[name] = delta
    return result


def verify_change(target: Path, mode: str) -> None:
    """Rebuild the witness from the durable checkpoint, then enforce the original policy."""
    record = a1.registered()
    receipt = read(target / "CHECKPOINT_RECEIPT.json")
    checkpoint = target / "checkpoint_last.pt"
    parent = a1.PARENT / "checkpoint_last.pt"
    if not receipt["complete_state"] or sha(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Boundary checkpoint is incomplete or changed")
    if sha(parent) != record["parent_sha256"]:
        raise ValueError("Boundary parent changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    initial = torch.load(parent, map_location="cpu", weights_only=False)
    if (
        payload["completed_updates"] != receipt["completed_updates"]
        or payload["identity_sha256"] != receipt["identity_sha256"]
        or payload["identity"]["mode"] != mode
        or payload["identity"]["parent_sha256"] != record["parent_sha256"]
    ):
        raise ValueError("Boundary checkpoint identity/cursor differs")
    evidence = {
        "updates": payload["completed_updates"],
        "maxabs": differences(payload["model_state_dict"], initial["model_state_dict"]),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "parent_sha256": record["parent_sha256"],
        "scope": "Durable endpoint versus shared registered initial parent; CPU only",
    }
    destination = target / "WEIGHT_CHANGE.json"
    if destination.exists():
        freeze(
            OUT / "previous_witnesses" / f"{target.name}_{sha(destination)}.json", read(destination)
        )
    write(destination, evidence)
    ORIGINAL_VERIFY(target, mode)


def admission() -> dict:
    """Bind this diagnostic-only wrapper to the original exact training admission."""
    runtime.admission()
    result = {
        "sources": {
            p: sha(ROOT / p)
            for p in (
                "operational/joint_measurement/adaptation_observed_run.py",
                "tests/unit/test_adaptation_boundary_witness.py",
            )
        },
        "training_admission_sha256": sha(runtime.OUT / "ADMISSION.json"),
        "training_changes": [],
        "changes": ["Fresh CPU weight witness at each durable scope", "Explicit child logs"],
    }
    freeze(OUT / "ADMISSION.json", result)
    return result


def run() -> None:
    """Reuse exclusive ownership and existing arm sequencing, with visible child errors."""
    admission()

    def launch(command: list[str], *, cwd: Path, creationflags: int) -> subprocess.Popen:
        checked = runtime.child_command(command)
        checked[4] = MODULE
        mode = checked[-1]
        with (
            (OUT / f"{mode}.stdout.log").open("ab", buffering=0) as stdout,
            (OUT / f"{mode}.stderr.log").open("ab", buffering=0) as stderr,
        ):
            return subprocess.Popen(
                checked, cwd=cwd, creationflags=creationflags, stdout=stdout, stderr=stderr
            )

    proxy = SimpleNamespace(Popen=launch, CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW)
    with (
        patch.object(runtime.base, "subprocess", proxy),
        patch.object(a1, "verify_change", verify_change),
    ):
        runtime.base.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "child", "admit", "verify"))
    parser.add_argument("--arm", choices=a1.MODES, default="FULL")
    args = parser.parse_args()
    admission()
    if args.action == "child":
        with patch.object(a1, "verify_change", verify_change):
            runtime.child(args.arm)
    elif args.action == "verify":
        for arm in a1.MODES:
            verify_change(a1.folder(arm), arm)
    elif args.action == "run":
        run()
