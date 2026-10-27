"""Fail-closed, additive provenance for exact OOF CPU prefetch execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from operational.final_refit.recipe import canonical_sha256, sha256
from operational.final_refit.runtime_migration import _STATE_FIELDS
from operational.final_refit.scientific_authority import load_config
from operational.rgb_port.accounting import atomic_write_bytes, atomic_write_json, read_json_shared

OPTIONS = {"prefetch_depth": 2, "decode_workers": 4, "decode_pending_bytes_per_role": 134217728}
ORIGIN_NAMES = (
    "checkpoint_last.pt",
    "CHECKPOINT_RECEIPT.json",
    "CHECKPOINT_POINTER.json",
    "UPDATE_JOURNAL.json",
)


def runtime_sources() -> tuple[Path, ...]:
    """Return the required complete additive execution/probe source closure."""
    root = Path(__file__).resolve().parents[1]
    return tuple(
        root / name
        for name in (
            "operational/scientific_oof_isolated.py",
            "operational/oof_runtime_binding.py",
            "operational/oof_runtime_overlay.py",
            "operational/oof_prefetch.py",
            "scripts/run_isolated_oof_stage.ps1",
            "scripts/probe_oof_prefetch.py",
            "scripts/profile_oof_loop.py",
        )
    )


def _sealed(path: Path) -> dict[str, Any]:
    value = read_json_shared(path)
    if canonical_sha256({k: v for k, v in value.items() if k != "payload_sha256"}) != value.get(
        "payload_sha256"
    ):
        raise ValueError(f"Runtime receipt digest differs: {path}")
    return value


def load_admission(run: Path) -> dict[str, Any]:
    """Validate the unchanged scientific authority and exact tested runtime files."""
    scientific = load_config(run)
    value = _sealed(run / "OOF_RUNTIME_ADMISSION.json")
    if (
        value.get("schema") != "oof_exact_prefetch_admission_v1"
        or value.get("status") != "PASS"
        or value.get("options") != OPTIONS
        or value.get("scientific_config_sha256") != sha256(run / "SCIENTIFIC_CONFIG.json")
    ):
        raise ValueError("Runtime optimization is not admitted")
    for category in ("source_sha256", "evidence_sha256"):
        bindings = value.get(category)
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError("Runtime admission lacks provenance")
        for path, digest in bindings.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Runtime admission binding changed: {path}")
    if set(value["source_sha256"]) != {str(p) for p in runtime_sources()}:
        raise ValueError("Runtime admission source closure is incomplete")
    proof = read_json_shared(Path(value["parity_report"]))
    expected_sources = {**scientific["source_sha256"], **value["source_sha256"]}
    if (
        value["evidence_sha256"].get(value["parity_report"]) != sha256(Path(value["parity_report"]))
        or proof.get("status") != "PASS"
        or proof.get("exact_training_state_parity") is not True
        or proof.get("schema") != "oof_prefetch_exact_state_probe_v1"
        or proof.get("exact_resume_parity") is not True
        or proof.get("original_checkpoint_unchanged") is not True
        or not isinstance(proof.get("technical_optimizer_updates"), int)
        or proof["technical_optimizer_updates"] < 1
        or proof.get("scientific_optimizer_updates") != 0
        or proof.get("sources_unchanged") is not True
        or proof.get("source_sha256") != expected_sources
        or proof.get("scientific_config_sha256") != value["scientific_config_sha256"]
    ):
        raise ValueError("Runtime lacks exact training-state parity")
    cases = {item["name"]: item for item in proof.get("cases", [])}
    required = {"baseline_A", "optimized_A", "optimized_B", "baseline_B", "optimized_resume"}
    if not required <= cases.keys():
        raise ValueError("Runtime proof lacks ABBA and resume cases")
    reference = cases["baseline_A"]["state_sha256"]
    if set(reference) != set(_STATE_FIELDS) or any(
        cases[name]["state_sha256"] != reference for name in required
    ):
        raise ValueError("Runtime full training state differs across cases")
    return value


def validate_binding(run: Path, directory: Path) -> dict[str, Any]:
    """Require the exact archived origin and admitted execution implementation."""
    admission = load_admission(run)
    value = _sealed(directory / "OOF_RUNTIME_BINDING.json")
    if (
        value.get("schema") != "oof_exact_prefetch_binding_v1"
        or value.get("admission_sha256") != sha256(run / "OOF_RUNTIME_ADMISSION.json")
        or value.get("admission_payload_sha256") != admission["payload_sha256"]
        or value.get("fit_directory") != str(directory.resolve())
    ):
        raise ValueError("Runtime fit binding differs")
    archive = directory.resolve() / "runtime_origin_v2"
    if set(value.get("origin_files_sha256", {})) != {str(archive / n) for n in ORIGIN_NAMES}:
        raise ValueError("Runtime origin archive is incomplete")
    for path, digest in value["origin_files_sha256"].items():
        if (
            not Path(path).resolve().is_relative_to(directory.resolve() / "runtime_origin_v2")
            or sha256(Path(path)) != digest
        ):
            raise ValueError("Runtime origin changed or escaped the fit")
    receipt = read_json_shared(archive / "CHECKPOINT_RECEIPT.json")
    pointer = read_json_shared(archive / "CHECKPOINT_POINTER.json")
    journal = read_json_shared(archive / "UPDATE_JOURNAL.json")
    digest = sha256(archive / "checkpoint_last.pt")
    if (
        value["origin_prior_binding"] is not None
        or digest != value["origin_checkpoint_sha256"]
        or journal["durable_updates"] != value["origin_updates"]
        or journal["pending_update_upper"] != 0
        or any(
            x["completed_updates"] != value["origin_updates"] for x in (receipt, pointer, journal)
        )
        or any(x.get("execution_binding_sha256") is not None for x in (receipt, pointer, journal))
        or any(x["checkpoint_sha256"] != digest for x in (receipt, pointer))
        or any(
            x["identity_sha256"] != value["identity_sha256"] for x in (receipt, pointer, journal)
        )
        or value.get("fit_id") != directory.name
        or value.get("rotation") != int(directory.parent.parent.name)
    ):
        raise ValueError("Runtime origin semantics are inconsistent")
    return value


def bind_fit(run: Path, directory: Path) -> dict[str, str | None]:
    """Admit a durable old checkpoint without rewriting any of its bytes."""
    path = directory / "OOF_RUNTIME_BINDING.json"
    if not path.exists():
        admission = load_admission(run)
        receipt = read_json_shared(directory / "CHECKPOINT_RECEIPT.json")
        pointer = read_json_shared(directory / "CHECKPOINT_POINTER.json")
        journal = read_json_shared(directory / "UPDATE_JOURNAL.json")
        checkpoint = directory / "checkpoint_last.pt"
        digest = sha256(checkpoint)
        if (
            receipt["completed_updates"] != journal["completed_updates"]
            or journal["completed_updates"] != journal["durable_updates"]
            or pointer["completed_updates"] != journal["completed_updates"]
            or journal["pending_update_upper"] != 0
            or receipt["checkpoint_sha256"] != digest
            or pointer["checkpoint_sha256"] != digest
            or receipt.get("execution_binding_sha256") is not None
            or pointer.get("execution_binding_sha256") is not None
            or journal.get("execution_binding_sha256") is not None
        ):
            raise ValueError("Runtime transition requires an exact durable unbound origin")
        archive = directory / "runtime_origin_v2"
        archive.mkdir(exist_ok=True)
        names = ORIGIN_NAMES
        for name in names:
            target = archive / name
            if target.exists():
                if sha256(target) != sha256(directory / name):
                    raise ValueError("A different runtime origin was already archived")
            else:
                atomic_write_bytes(target, (directory / name).read_bytes())
        value = dict(
            schema="oof_exact_prefetch_binding_v1",
            fit_directory=str(directory.resolve()),
            fit_id=directory.name,
            rotation=int(directory.parent.parent.name),
            admission_sha256=sha256(run / "OOF_RUNTIME_ADMISSION.json"),
            admission_payload_sha256=admission["payload_sha256"],
            origin_checkpoint_sha256=digest,
            origin_prior_binding=None,
            origin_updates=receipt["completed_updates"],
            identity_sha256=receipt["identity_sha256"],
            origin_files_sha256={str((archive / n).resolve()): sha256(archive / n) for n in names},
        )
        value["payload_sha256"] = canonical_sha256(value)
        atomic_write_json(path, value)
    value = validate_binding(run, directory)
    return dict(
        execution_binding_sha256=sha256(path),
        execution_binding_origin_checkpoint_sha256=value["origin_checkpoint_sha256"],
        execution_binding_origin_prior_sha256=value["origin_prior_binding"],
    )


def verify_runtime_endpoint(path: Path, admission: dict[str, Any]) -> None:
    """Add runtime verification to the original scientific endpoint verifier."""
    import torch

    directory = path.parent
    receipt = read_json_shared(directory / "CHECKPOINT_RECEIPT.json")
    binding = receipt.get("execution_binding_sha256")
    binding_path = directory / "OOF_RUNTIME_BINDING.json"
    if binding is None and not binding_path.exists():
        if admission["fit_id"] in ("R_A5", "R_C2F"):
            raise ValueError("RGB endpoint lacks its admitted runtime binding")
        return
    run = directory.parents[3]
    value = validate_binding(run, directory)
    digest = sha256(binding_path)
    if (
        binding != digest
        or value["identity_sha256"] != receipt["identity_sha256"]
        or admission["fit_id"] != value["fit_id"]
        or admission["rotation"] != value["rotation"]
        or receipt["completed_updates"] < value["origin_updates"]
    ):
        raise ValueError("Endpoint runtime identity/binding differs")
    stored = torch.load(Path(admission["checkpoint_path"]), map_location="cpu", weights_only=False)
    for item in [
        stored,
        read_json_shared(directory / "CHECKPOINT_POINTER.json"),
        read_json_shared(directory / "UPDATE_JOURNAL.json"),
    ]:
        if item.get("execution_binding_sha256") != digest:
            raise ValueError("Endpoint runtime binding is incomplete")
