"""Admit an exact paused Windows event checkpoint on Linux without identity rewriting."""

# ruff: noqa: ANN401 -- frozen scientific APIs are patched without changing signatures.

from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from operational.final_refit.recipe import canonical_sha256
from operational.rgb_port.accounting import atomic_write_json, read_json_shared

from .inventory import digest

BINDING_SCHEMA = "ejepa_event_cross_hardware_execution_binding_v1"
ORIGIN_NAMES = (
    "CHECKPOINT_POINTER.json",
    "CHECKPOINT_RECEIPT.json",
    "UPDATE_JOURNAL.json",
)


def _portable_source_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only physical path spellings; retain every scientific/content field."""
    result = dict(value)
    result.pop("identity_sha256", None)
    storage = result.get("storage_source_identity")
    if not isinstance(storage, dict):
        raise ValueError("Event source identity lacks storage roles")
    normalized = {}
    for role, raw in storage.items():
        if not isinstance(raw, dict):
            raise ValueError("Event storage role identity is malformed")
        item = dict(raw)
        for key in ("role_manifest_path", "event_source_root"):
            path = item.pop(key, None)
            if not isinstance(path, str) or not path:
                raise ValueError(f"Event storage identity lacks {key}")
        item.pop("identity_sha256", None)
        normalized[role] = item
    result["storage_source_identity"] = normalized
    return result


def _origin(directory: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    pointer = read_json_shared(directory / "CHECKPOINT_POINTER.json")
    receipt = read_json_shared(directory / "CHECKPOINT_RECEIPT.json")
    journal = read_json_shared(directory / "UPDATE_JOURNAL.json")
    version = directory / "checkpoint_versions" / str(pointer["version"])
    checksum = digest(version)
    if not (
        pointer.get("schema") == "rgb_port_checkpoint_pointer_v1"
        and receipt.get("schema") == "rgb_port_checkpoint_receipt_v1"
        and journal.get("schema") == "rgb_port_update_journal_v1"
        and pointer.get("completed_updates")
        == receipt.get("completed_updates")
        == journal.get("completed_updates")
        == journal.get("durable_updates")
        and int(journal.get("pending_update_upper", -1)) == 0
        and int(journal.get("recovery_upper", -1)) == 0
        and int(receipt.get("accumulation_index", -1)) == 0
        and pointer.get("checkpoint_sha256") == receipt.get("checkpoint_sha256") == checksum
        and pointer.get("identity_sha256")
        == receipt.get("identity_sha256")
        == journal.get("identity_sha256")
        and all(
            value.get("execution_binding_sha256") is None for value in (pointer, receipt, journal)
        )
    ):
        raise ValueError("Imported event checkpoint is not one exact durable unbound transaction")
    return version, pointer, receipt, journal


def validate_origin_identity(
    identity: Mapping[str, Any], source: object, recipe: object, *, fit: str
) -> None:
    """Validate scientific fields directly; environment remains the recorded Windows origin."""
    source_identity = dict(source.identity)
    recipe_identity = dict(recipe.identity)
    rotation = int(source_identity["rotation"])
    if not (
        identity.get("schema") == "final_refit_fold_producer_identity_v1"
        and identity.get("fit_id") == fit
        and identity.get("rotation") == rotation
        and identity.get("technical") is False
        and isinstance(identity.get("source_identity"), dict)
        and _portable_source_identity(identity["source_identity"])
        == _portable_source_identity(source_identity)
        and identity.get("producer_recipe") == recipe_identity
        and identity.get("recipe_identity_sha256") == recipe.identity_sha256
        and int(identity.get("endpoint_updates", -1)) == int(recipe.updates)
        and int(identity.get("checkpoint_interval_updates", -1)) == 100
        and identity.get("initialization") == "fresh_seed7_no_checkpoint_import"
        and identity.get("device") == "cuda"
    ):
        raise ValueError(
            "Paused checkpoint scientific identity differs from this event fold/recipe"
        )
    environment = identity.get("environment")
    if not isinstance(environment, dict) or environment.get("python") != "3.11.15":
        raise ValueError("Paused checkpoint lacks its exact Windows environment identity")
    payload = {k: v for k, v in identity.items() if k != "identity_sha256"}
    if canonical_sha256(payload) != identity.get("identity_sha256"):
        raise ValueError("Paused checkpoint identity digest differs")


def create_binding(
    directory: Path,
    source: object,
    recipe: object,
    *,
    fit: str,
    assignment: Path,
    pause_receipt: Path,
    technical: bool,
) -> dict[str, Any]:
    """Bind an immutable origin to Linux execution, preserving its embedded identity bytes."""
    import torch

    directory = directory.resolve(strict=True)
    path = directory / "EVENT_EXECUTION_BINDING.json"
    if path.exists():
        value = read_json_shared(path)
        claimed = dict(value)
        payload_sha256 = claimed.pop("payload_sha256", None)
        if (
            value.get("schema") != BINDING_SCHEMA
            or payload_sha256 != canonical_sha256(claimed)
            or value.get("technical") is not technical
            or value.get("rotation") != int(source.identity["rotation"])
            or value.get("fit") != fit
            or value.get("assignment_sha256") != digest(assignment.resolve(strict=True))
            or value.get("pause_receipt_sha256") != digest(pause_receipt.resolve(strict=True))
            or value.get("identity_bytes_rewritten") is not False
            or value.get("cross_hardware_bitwise_continuity_claimed") is not False
        ):
            raise ValueError("Existing event execution binding differs or is corrupt")
        origin = Path(str(value["origin_checkpoint"]))
        if not origin.is_file() or digest(origin) != value.get("origin_checkpoint_sha256"):
            raise ValueError("Existing event binding origin checkpoint differs")
        import torch

        saved = torch.load(origin, map_location="cpu", weights_only=False)
        identity = saved.get("identity")
        if not isinstance(identity, dict) or saved.get("identity_sha256") != value.get(
            "origin_identity_sha256"
        ):
            raise ValueError("Existing event binding origin identity differs")
        validate_origin_identity(identity, source, recipe, fit=fit)
        return value
    version, pointer, _, _ = _origin(directory)
    saved = torch.load(version, map_location="cpu", weights_only=False)
    identity = saved.get("identity")
    if not isinstance(identity, dict) or saved.get("identity_sha256") != pointer["identity_sha256"]:
        raise ValueError("Paused checkpoint payload identity differs")
    validate_origin_identity(identity, source, recipe, fit=fit)
    pause = read_json_shared(pause_receipt.resolve(strict=True))
    if not (
        pause.get("status") == "PAUSED_DURABLY_EVENT_POD_HANDOFF_PENDING"
        and pause.get("rotation") == source.identity["rotation"]
        and pause.get("fit") == fit
        and pause.get("completed_updates") == pointer["completed_updates"]
        and pause.get("checkpoint_sha256") == pointer["checkpoint_sha256"]
        and pause.get("updates_lost") == 0
        and pause.get("cross_hardware_numerical_continuity_claimed") is False
    ):
        raise ValueError("Local event handoff pause receipt differs from the imported checkpoint")
    value = {
        "schema": BINDING_SCHEMA,
        "status": "TECHNICAL" if technical else "ADMITTED_PENDING_PROBE",
        "technical": technical,
        "rotation": int(source.identity["rotation"]),
        "fit": fit,
        "origin_checkpoint": str(version),
        "origin_checkpoint_sha256": pointer["checkpoint_sha256"],
        "origin_updates": int(pointer["completed_updates"]),
        "origin_identity_sha256": pointer["identity_sha256"],
        "origin_execution_binding_sha256": None,
        "pause_receipt": str(pause_receipt.resolve()),
        "pause_receipt_sha256": digest(pause_receipt),
        "assignment": str(assignment.resolve(strict=True)),
        "assignment_sha256": digest(assignment),
        "target_environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "identity_bytes_rewritten": False,
        "optimizer_or_scheduler_state_reinitialized": False,
        "sampler_or_rng_state_reinitialized": False,
        "cross_hardware_bitwise_continuity_claimed": False,
        "restart_probe_required": True,
    }
    value["payload_sha256"] = canonical_sha256(value)
    atomic_write_json(path, value)
    return value


def _binding_kwargs(path: Path, value: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "execution_binding_sha256": digest(path),
        "execution_binding_origin_checkpoint_sha256": str(value["origin_checkpoint_sha256"]),
        "execution_binding_origin_prior_sha256": None,
    }


@contextmanager
def activated(
    science: Any,
    directory: Path,
    source: object,
    recipe: object,
    *,
    fit: str,
    assignment: Path,
    pause_receipt: Path,
    technical: bool,
) -> Iterator[dict[str, Any]]:
    """Patch only identity construction and checkpoint binding for the imported origin."""
    value = create_binding(
        directory,
        source,
        recipe,
        fit=fit,
        assignment=assignment,
        pause_receipt=pause_receipt,
        technical=technical,
    )
    binding_path = directory / "EVENT_EXECUTION_BINDING.json"
    import torch

    saved = torch.load(
        Path(str(value["origin_checkpoint"])),
        map_location="cpu",
        weights_only=False,
    )
    origin_identity = dict(saved["identity"])
    original_identity = science.identity_for
    original_authority = science.execution_binding
    original_checkpoint = science.ProducerCheckpoint
    kwargs = _binding_kwargs(binding_path, value)

    def identity(*args: Any, **named: Any) -> dict[str, Any]:
        current = original_identity(*args, **named)
        validate_origin_identity(origin_identity, source, recipe, fit=fit)
        for field in (
            "fit_id",
            "rotation",
            "producer_recipe",
            "recipe_identity_sha256",
            "fold_contract_sha256",
            "rotation_manifest_sha256",
            "training_cache_sha256",
            "training_sample_token_sha256",
            "train_group_ids",
            "generation_group_ids",
            "endpoint_updates",
            "checkpoint_interval_updates",
        ):
            if current.get(field) != origin_identity.get(field):
                raise ValueError(f"Cloud scientific identity changed field {field}")
        if _portable_source_identity(current["source_identity"]) != _portable_source_identity(
            origin_identity["source_identity"]
        ):
            raise ValueError("Cloud scientific source content identity changed")
        return dict(origin_identity)

    class BoundCheckpoint(original_checkpoint):
        def __init__(self, *args: object, **named: object) -> None:
            injected = dict(named)
            injected.update(kwargs)
            super().__init__(*args, **injected)

    def authority(_run: Path) -> dict[str, Any]:
        # `train` adds this after calling identity_for. Preserve the exact origin
        # authority so checkpoint identity bytes are never rewritten.
        return dict(origin_identity["execution_authority"])

    science.identity_for = identity
    science.execution_binding = authority
    science.ProducerCheckpoint = BoundCheckpoint
    try:
        yield value
    finally:
        science.identity_for = original_identity
        science.execution_binding = original_authority
        science.ProducerCheckpoint = original_checkpoint


def install_handoff(archive_root: Path, run: Path) -> dict[str, Any]:
    """Install a previously extracted, verified handoff into an empty cloud fit directory."""
    archive_root, run = archive_root.resolve(strict=True), run.resolve()
    manifest = read_json_shared(archive_root / "HANDOFF_MANIFEST.json")
    if manifest.get("schema") != "ejepa_event_checkpoint_handoff_v1":
        raise ValueError("Event handoff manifest schema differs")
    target = run / "rotations/2/fits/E_A5"
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Cloud E_A5/R2 fit is not empty")
    target.mkdir(parents=True, exist_ok=True)
    for relative, binding in manifest["files"].items():
        if not relative.startswith("rotations/2/fits/E_A5/"):
            continue
        suffix = Path(relative).relative_to("rotations/2/fits/E_A5")
        incoming = archive_root / relative
        if incoming.stat().st_size != binding["bytes"] or digest(incoming) != binding["sha256"]:
            raise ValueError(f"Handoff member differs: {relative}")
        destination = target / suffix
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(incoming, destination)
    version = target / "checkpoint_versions" / f"checkpoint_{manifest['completed_updates']:06d}.pt"
    alias = target / "checkpoint_last.pt"
    try:
        os.link(version, alias)
    except OSError:
        shutil.copyfile(version, alias)
    if digest(alias) != manifest["checkpoint_sha256"]:
        raise ValueError("Installed event handoff alias differs")
    return manifest
