"""Bounded checkpoint parity probe; all probe identities are explicitly technical."""

# ruff: noqa: ANN401 -- frozen training modules and heterogeneous checkpoint state.

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .inventory import digest

STATE_FIELDS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "sampler_generator_state",
    "torch_rng_state",
    "cuda_rng_state_all",
    "numpy_random_state",
    "python_random_state",
    "cursor",
    "completed_updates",
    "recovery_upper",
    "accumulation_index",
)


def state_digest(value: Any) -> str:
    """Hash numerical checkpoint state without pickle storage identities."""
    import numpy as np
    import torch

    result = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            visit(item.detach().cpu().contiguous().numpy())
        elif isinstance(item, np.ndarray):
            result.update(f"array:{item.dtype}:{item.shape}:".encode())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            result.update(b"dict:")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child)
        elif isinstance(item, (int, float, str, bool)) or item is None:
            result.update(f"{type(item).__name__}:".encode())
            result.update(json.dumps(item, allow_nan=False).encode())
        elif isinstance(item, np.generic):
            visit(item.item())
        else:
            raise TypeError(f"Unsupported checkpoint field: {type(item)}")

    visit(value)
    return result.hexdigest()


def probe(
    source: Any,
    science: Any,
    config: dict,
    run: Path,
    fit: str,
    monitor: Any,
    migration_sha256: str,
) -> dict:
    """Compare four uninterrupted updates against two plus restored two: eight total."""
    import torch

    from operational.final_refit.recipe import canonical_sha256

    probe_root = run / "technical_resume_probe" / fit
    if probe_root.exists():
        raise FileExistsError("Probe already exists: inspect its result; do not silently repeat")
    original_identity = science.identity_for

    def technical_identity(*args: Any, **kwargs: Any) -> dict:
        identity = original_identity(*args, **kwargs)
        identity.pop("identity_sha256")
        identity.update(
            technical=True,
            scientific_endpoint=False,
            cloud_migration_sha256=migration_sha256,
            probe="four_vs_two_plus_two_no_endpoint_admission",
        )
        identity["identity_sha256"] = canonical_sha256(identity)
        return identity

    science.identity_for = technical_identity
    completed = 0
    try:
        outputs = {}
        for name, counts in (("uninterrupted", (4,)), ("restored", (2, 2))):
            destination = probe_root / name
            destination.mkdir(parents=True)
            shutil.copyfile(run / "SCIENTIFIC_CONFIG.json", destination / "SCIENTIFIC_CONFIG.json")
            for count in counts:
                receipt = science.train(
                    source=source,
                    fit=fit,
                    config=Path(config["config"]),
                    contract=Path(config["fold_contract"]),
                    run=destination,
                    monitor=monitor,
                    device="cuda",
                    max_updates=count,
                )
                if receipt["status"] == "PAUSED_RESOURCE":
                    raise RuntimeError("Resource pause during technical probe; result not admitted")
                completed += count
                torch.cuda.empty_cache()
            checkpoint = Path(receipt["checkpoint_path"])
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["identity"]["technical"] is not True or saved["completed_updates"] != 4:
                raise ValueError("Probe checkpoint identity is not a four-update technical run")
            outputs[name] = {
                "checkpoint_sha256": digest(checkpoint),
                "state": {key: state_digest(saved[key]) for key in STATE_FIELDS},
            }
        equal = outputs["uninterrupted"]["state"] == outputs["restored"]["state"]
        report = {
            "schema": "ejepa_linux_resume_probe_v1",
            "status": "PASS" if equal else "FAIL",
            "technical_optimizer_updates": completed,
            "scientific_optimizer_updates": 0,
            "migration_binding_sha256": migration_sha256,
            "fit": fit,
            "rotation": source.identity["rotation"],
            "cases": outputs,
            "same_process_reconstruction_resume": True,
            "abrupt_power_loss_tested": False,
            "cross_hardware_bitwise_parity_claimed": False,
        }
        (probe_root / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
        if not equal:
            raise ValueError("Linux checkpoint resume changed numerical state")
        return report
    finally:
        science.identity_for = original_identity
