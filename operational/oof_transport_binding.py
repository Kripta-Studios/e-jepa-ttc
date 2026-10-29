"""Additive, versioned authority for caching constant transport masks."""

from pathlib import Path
from typing import Any

from operational.final_refit.recipe import canonical_sha256, sha256
from operational.final_refit.runtime_migration import _STATE_FIELDS
from operational.oof_runtime_binding import (
    ORIGIN_NAMES,
    _sealed,
)
from operational.oof_runtime_binding import (
    bind_fit as parent_bind,
)
from operational.oof_runtime_binding import (
    load_admission as parent_admission,
)
from operational.oof_runtime_binding import (
    runtime_sources as parent_sources,
)
from operational.oof_runtime_binding import (
    validate_binding as parent_validate,
)
from operational.oof_runtime_binding import (
    verify_runtime_endpoint as parent_verify,
)
from operational.rgb_port.accounting import atomic_write_bytes, atomic_write_json, read_json_shared


def runtime_sources() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[1]
    return (
        *parent_sources(),
        *(
            root / name
            for name in (
                "operational/oof_transport_cache.py",
                "operational/oof_transport_binding.py",
                "operational/scientific_oof_transport.py",
                "scripts/run_transport_oof_stage.ps1",
                "scripts/probe_oof_transport_steady.py",
            )
        ),
    )


def load_admission(run: Path) -> dict[str, Any]:
    """Check the original authority and the new proof and source closure."""
    parent_admission(run)
    record = _sealed(run / "OOF_TRANSPORT_ADMISSION.json")
    expected = {str(p): sha256(p) for p in runtime_sources()}
    if (
        record.get("schema") != "oof_transport_admission_v1"
        or record.get("status") != "PASS"
        or record.get("options")
        != {"max_cache_bytes": 67108864, "max_entries": 16, "fits": ["R_C2F"]}
        or record.get("source_sha256") != expected
        or record.get("scientific_config_sha256") != sha256(run / "SCIENTIFIC_CONFIG.json")
        or record.get("tests_sha256") != sha256(run / "OOF_TRANSPORT_TESTS.xml")
    ):
        raise ValueError("Transport runtime authority differs")
    proof_path = Path(record["parity_report"])
    if sha256(proof_path) != record["proof_sha256"]:
        raise ValueError("Transport parity evidence changed")
    proof = read_json_shared(proof_path)
    if (
        proof.get("status") != "PASS"
        or proof.get("schema") != "oof_transport_steady_probe_v1"
        or proof.get("shared_source") is not True
        or proof.get("warmup_updates", 0) < 32
        or proof.get("exact_training_state_parity") is not True
        or proof.get("exact_resume_parity") is not True
        or proof.get("sources_unchanged") is not True
        or proof.get("original_checkpoint_unchanged") is not True
        or proof.get("scientific_optimizer_updates") != 0
        or proof.get("technical_optimizer_updates", 0) <= 0
        or proof.get("scientific_config_sha256") != record["scientific_config_sha256"]
    ):
        raise ValueError("Transport runtime lacks exact-state evidence")
    for name in (
        "operational/oof_transport_cache.py",
        "scripts/probe_oof_transport_steady.py",
    ):
        path = Path(__file__).resolve().parents[1] / name
        if proof["source_sha256"].get(str(path)) != sha256(path):
            raise ValueError("Proof did not execute the current implementation")
    for path, digest in proof["source_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Proven source changed")
    cases = {c["name"]: c for c in proof["cases"]}
    reference = cases["baseline_A"]["state_sha256"]
    if set(reference) != set(_STATE_FIELDS):
        raise ValueError("Incomplete transport state comparison")
    for name in ("optimized_A", "optimized_B", "baseline_B", "optimized_resume"):
        if cases[name]["state_sha256"] != reference:
            raise ValueError("Training states differ")
    if sum(cases[n]["seconds_per_update"] for n in ("optimized_A", "optimized_B")) >= sum(
        cases[n]["seconds_per_update"] for n in ("baseline_A", "baseline_B")
    ):
        raise ValueError("Transport cache did not improve paired execution time")
    return record


def validate_binding(run: Path, directory: Path) -> dict[str, Any]:
    """Preserve the preceding runtime origin as an explicit parent."""
    load_admission(run)
    parent = parent_validate(run, directory)
    record = _sealed(directory / "OOF_TRANSPORT_BINDING.json")
    if (
        record.get("schema") != "oof_transport_binding_v1"
        or record["fit_directory"] != str(directory.resolve())
        or record["fit_id"] != "R_C2F"
        or record["identity_sha256"] != parent["identity_sha256"]
        or record["parent_binding_sha256"] != sha256(directory / "OOF_RUNTIME_BINDING.json")
        or record["admission_sha256"] != sha256(run / "OOF_TRANSPORT_ADMISSION.json")
    ):
        raise ValueError("Transport binding lineage differs")
    archive = directory.resolve() / "transport_origin_v3"
    if set(record["origin_files_sha256"]) != {str(archive / n) for n in ORIGIN_NAMES}:
        raise ValueError("Transport origin archive incomplete")
    for path, digest in record["origin_files_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Transport origin changed")
    receipt = read_json_shared(archive / "CHECKPOINT_RECEIPT.json")
    pointer = read_json_shared(archive / "CHECKPOINT_POINTER.json")
    journal = read_json_shared(archive / "UPDATE_JOURNAL.json")
    if (
        receipt["checkpoint_sha256"] != record["origin_checkpoint_sha256"]
        or pointer["checkpoint_sha256"] != record["origin_checkpoint_sha256"]
        or sha256(archive / "checkpoint_last.pt") != record["origin_checkpoint_sha256"]
        or journal["pending_update_upper"] != 0
        or any(
            x["completed_updates"] != record["origin_updates"] for x in (receipt, pointer, journal)
        )
        or journal["durable_updates"] != record["origin_updates"]
        or any(
            x.get("execution_binding_sha256") != record["origin_prior_binding"]
            for x in (receipt, pointer, journal)
        )
        or receipt["identity_sha256"] != record["identity_sha256"]
    ):
        raise ValueError("Transport origin counters or bindings differ")
    prior = record["origin_prior_binding"]
    if prior != record["parent_binding_sha256"]:
        if (
            prior is not None
            or record["origin_checkpoint_sha256"] != parent["origin_checkpoint_sha256"]
        ):
            raise ValueError("Unrecognized transport origin parent")
    return record


def bind_fit(run: Path, directory: Path) -> dict[str, Any]:
    """Migrate only R_C2F, accepting only its exact durable prior checkpoint."""
    if directory.name != "R_C2F":
        return parent_bind(run, directory)
    load_admission(run)
    parent_bind(run, directory)
    path = directory / "OOF_TRANSPORT_BINDING.json"
    if not path.exists():
        receipt = read_json_shared(directory / "CHECKPOINT_RECEIPT.json")
        journal = read_json_shared(directory / "UPDATE_JOURNAL.json")
        pointer = read_json_shared(directory / "CHECKPOINT_POINTER.json")
        digest = sha256(directory / "checkpoint_last.pt")
        if (
            journal["pending_update_upper"] != 0
            or journal["completed_updates"] != journal["durable_updates"]
            or receipt["completed_updates"] != journal["completed_updates"]
            or pointer["completed_updates"] != journal["completed_updates"]
            or receipt["checkpoint_sha256"] != digest
            or pointer["checkpoint_sha256"] != digest
        ):
            raise ValueError("Transport migration requires a durable origin")
        archive = directory / "transport_origin_v3"
        archive.mkdir(exist_ok=True)
        for name in ORIGIN_NAMES:
            target = archive / name
            if target.exists() and sha256(target) != sha256(directory / name):
                raise ValueError("Different transport origin already archived")
            if not target.exists():
                atomic_write_bytes(target, (directory / name).read_bytes())
        value = dict(
            schema="oof_transport_binding_v1",
            fit_directory=str(directory.resolve()),
            fit_id=directory.name,
            identity_sha256=receipt["identity_sha256"],
            parent_binding_sha256=sha256(directory / "OOF_RUNTIME_BINDING.json"),
            admission_sha256=sha256(run / "OOF_TRANSPORT_ADMISSION.json"),
            origin_checkpoint_sha256=digest,
            origin_updates=receipt["completed_updates"],
            origin_prior_binding=receipt.get("execution_binding_sha256"),
            origin_files_sha256={
                str((archive / n).resolve()): sha256(archive / n) for n in ORIGIN_NAMES
            },
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
    """Verify final endpoints without invalidating completed preceding-runtime fits."""
    import torch

    directory = path.parent
    if not (directory / "OOF_TRANSPORT_BINDING.json").exists():
        parent_verify(path, admission)
        return
    value = validate_binding(directory.parents[3], directory)
    digest = sha256(directory / "OOF_TRANSPORT_BINDING.json")
    receipt = read_json_shared(directory / "CHECKPOINT_RECEIPT.json")
    if (
        admission["fit_id"] != value["fit_id"]
        or receipt["identity_sha256"] != value["identity_sha256"]
        or receipt["completed_updates"] < value["origin_updates"]
    ):
        raise ValueError("Transport endpoint identity differs")
    saved = torch.load(Path(admission["checkpoint_path"]), map_location="cpu", weights_only=False)
    for item in (
        receipt,
        saved,
        read_json_shared(directory / "CHECKPOINT_POINTER.json"),
        read_json_shared(directory / "UPDATE_JOURNAL.json"),
    ):
        if item.get("execution_binding_sha256") != digest:
            raise ValueError("Transport endpoint execution binding differs")
