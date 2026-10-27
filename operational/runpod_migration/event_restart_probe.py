"""Eight-update Linux restart proof beginning at the exact paused Windows state."""

# ruff: noqa: ANN401 -- frozen scientific runner is passed through unchanged.

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json

from .event_handoff import ORIGIN_NAMES, activated
from .inventory import digest
from .probe import STATE_FIELDS, state_digest


def _clone_origin(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    pointer = json.loads((source / "CHECKPOINT_POINTER.json").read_text(encoding="utf-8"))
    version_name = str(pointer["version"])
    (target / "checkpoint_versions").mkdir()
    shutil.copyfile(
        source / "checkpoint_versions" / version_name,
        target / "checkpoint_versions" / version_name,
    )
    shutil.copyfile(source / "checkpoint_versions" / version_name, target / "checkpoint_last.pt")
    for name in (*ORIGIN_NAMES, "PROGRESS.json"):
        shutil.copyfile(source / name, target / name)


def probe(
    science: Any,
    source: object,
    recipe: object,
    config: dict[str, Any],
    run: Path,
    *,
    fit: str,
    assignment: Path,
    pause_receipt: Path,
    monitor: object,
) -> dict[str, Any]:
    """Compare origin+4 with origin+2+restore+2; exactly eight technical updates."""
    import torch

    if int(source.identity["rotation"]) != 2 or fit != "E_A5":
        raise ValueError("Imported event restart proof is restricted to E_A5/R2")
    origin = run / "rotations/2/fits/E_A5"
    origin_pointer = json.loads((origin / "CHECKPOINT_POINTER.json").read_text(encoding="utf-8"))
    origin_updates = int(origin_pointer["completed_updates"])
    root = run / "technical_event_restart_probe/r2/E_A5"
    if root.exists():
        raise FileExistsError("Event restart probe already exists; never repeat silently")
    outputs: dict[str, Any] = {}
    completed = 0
    for name, counts in (("uninterrupted", (4,)), ("restored", (2, 2))):
        case_run = root / name
        case_run.mkdir(parents=True)
        shutil.copyfile(run / "SCIENTIFIC_CONFIG.json", case_run / "SCIENTIFIC_CONFIG.json")
        fit_directory = case_run / "rotations/2/fits/E_A5"
        _clone_origin(origin, fit_directory)
        with activated(
            science,
            fit_directory,
            source,
            recipe,
            fit=fit,
            assignment=assignment,
            pause_receipt=pause_receipt,
            technical=True,
        ):
            for count in counts:
                receipt = science.train(
                    source=source,
                    fit=fit,
                    config=Path(config["config"]),
                    contract=Path(config["fold_contract"]),
                    run=case_run,
                    monitor=monitor,
                    device="cuda",
                    max_updates=count,
                )
                if receipt["status"] == "PAUSED_RESOURCE":
                    raise RuntimeError("Resource pause during event restart proof")
                completed += count
                torch.cuda.empty_cache()
        checkpoint = Path(receipt["checkpoint_path"])
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(saved["completed_updates"]) != origin_updates + 4:
            raise ValueError("Event restart probe escaped its four-update budget")
        outputs[name] = {
            "checkpoint_sha256": digest(checkpoint),
            "state": {field: state_digest(saved[field]) for field in STATE_FIELDS},
            "execution_binding_sha256": saved.get("execution_binding_sha256"),
        }
    equal = outputs["uninterrupted"]["state"] == outputs["restored"]["state"]
    result = {
        "schema": "ejepa_linux_resume_probe_v1",
        "status": "PASS" if equal else "FAIL",
        "rotation": 2,
        "fit": fit,
        "origin_updates": origin_updates,
        "origin_checkpoint_sha256": digest(
            origin / "checkpoint_versions" / str(origin_pointer["version"])
        ),
        "technical_optimizer_updates": completed,
        "scientific_optimizer_updates": 0,
        "cases": outputs,
        "same_linux_hardware_restart_exact": equal,
        "cross_hardware_bitwise_parity_claimed": False,
        "windows_identity_rewritten": False,
    }
    result_path = root / "RESULT.json"
    atomic_write_json(result_path, result)
    if not equal or completed != 8:
        raise ValueError("Imported event restart proof failed")
    return result
