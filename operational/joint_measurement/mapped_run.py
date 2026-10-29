"""Run the existing primary queue with an explicitly bound, QA-gated I/O adapter."""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import psutil

from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from . import run as primary
from . import train
from .common import ROOT, RUN
from .mapped_data import mapped_root
from .mapped_runtime import MappedSensorDataset, admit_mapped


def validate_qa(result: dict, expected_fits: list[str]) -> None:
    """Refuse incomplete, unequal or slower diagnostic evidence."""
    if (
        result.get("status") != "PASS"
        or result.get("performance_gate_pass") is not True
        or result.get("optimizer_updates") != 80
        or set(result.get("checks", {})) != set(expected_fits)
    ):
        raise ValueError("Mapped runtime requires complete exact-resume and speed evidence")
    ratios = []
    expected = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "numpy_random_state",
        "python_random_state",
        "cursor",
    }
    for check in result["checks"].values():
        if check.get("status") != "PASS" or set(check.get("exact", {})) != expected:
            raise ValueError("Mapped runtime exact-state coverage is incomplete")
        if not all(value is True for value in check["exact"].values()):
            raise ValueError("Mapped runtime changed training state")
        ratios.append(check["mapped_to_serial_ratio"])
    if not all(0 < ratio <= 0.9 for ratio in ratios) or sum(ratios) / len(ratios) > 0.8:
        raise ValueError("Mapped runtime speed gate is not supported by its measurements")


def assert_no_primary_writer() -> None:
    """Exclude this launcher's own ancestors; reject every other live queue writer."""
    own = psutil.Process()
    excluded = {own.pid, *(p.pid for p in own.parents())}
    modules = {"operational.joint_measurement.run", "operational.joint_measurement.mapped_run"}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid in excluded or "python" not in (process.info["name"] or "").lower():
            continue
        if modules.intersection(process.info["cmdline"] or []):
            try:
                if Path(process.cwd()).resolve() == ROOT:
                    raise RuntimeError(f"Another primary writer remains live: {process.pid}")
            except psutil.NoSuchProcess:
                continue


def registration() -> dict:
    """Extend runtime provenance without rewriting existing scientific identities."""
    diagnostic = RUN / "diagnostics/mapped_runtime"
    evidence = read(diagnostic / "COMPLETE.json")
    recipe = train.config()
    validate_qa(evidence, recipe["fits"])
    registered = read(diagnostic / "REGISTRATION.json")
    original = train.sources()
    if (
        evidence["registration_sha256"] != sha(diagnostic / "REGISTRATION.json")
        or evidence["input_parity_sha256"] != sha(diagnostic / "INPUT_PARITY.json")
        or evidence["scientific_source_hash"] != digest(original)
        or registered["scientific_sources"] != original
    ):
        raise ValueError("Mapped QA no longer describes the admitted scientific sources")
    for path, expected in registered["runtime_sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Mapped adapter changed since its diagnostic")
    if registered["new_cache_binding"] != sha(mapped_root() / "BINDING.json"):
        raise ValueError("Uncompressed input genealogy changed")
    transition = RUN / "scheduling/MAPPED_INPUT/STOP_RESULT.json"
    stopped = read(transition)
    if stopped["status"] not in {"STOPPED_AT_DURABLE_BOUNDARY", "EXITED_AT_DURABLE_BOUNDARY"}:
        raise ValueError("Original writer did not stop at a verified durable boundary")
    paths = ["mapped_run.py", "mapped_runtime.py", "mapped_data.py", "run.py", "schedule.py"]
    runtime_sources = {
        f"operational/joint_measurement/{name}": sha(ROOT / "operational/joint_measurement" / name)
        for name in paths
    }
    train.committed_sources(runtime_sources)
    record = {
        "schema": "v18_explicit_input_runtime_extension_v1",
        "scientific_identity_policy": (
            "Preserve existing identities; append runtime source and checkpoint segment bindings"
        ),
        "mechanism": (
            "Explicit process-local replacement of train.SensorDataset with tested subclass; "
            "same inherited native batch collation"
        ),
        "scientific_source_hash": digest(original),
        "runtime_sources": runtime_sources,
        "qa_complete_sha256": sha(diagnostic / "COMPLETE.json"),
        "cache_complete_sha256": sha(mapped_root() / "COMPLETE.json"),
        "transition_sha256": sha(transition),
        "recipe_changed": False,
        "endpoint_changed": False,
        "model_graph_changed": False,
        "data_values_changed": False,
        "no_retraining_or_reset": True,
        "scope": "P input transport only; primary inference remains the original sealed evaluator",
    }
    freeze(RUN / "runtime/MAPPED_INPUT.json", record)
    return record


def run(*, once: bool = False) -> None:
    """Reuse the single existing queue; write explicit provenance for every scope."""
    assert_no_primary_writer()
    binding = registration()
    started = time.perf_counter()
    reader = admit_mapped()
    write(
        RUN / "runtime/admissions" / f"{os.getpid()}_{time.time_ns()}.json",
        {
            "utc": utc(),
            "cpu_byte_admission_seconds": time.perf_counter() - started,
            "gpu_scope": False,
            "runtime_binding": digest(binding),
            "packs": len(reader.verified),
        },
    )

    def fit_with_provenance(fit_id: str) -> dict:
        registration()
        folder = RUN / "fits" / fit_id
        receipt_path = folder / "CHECKPOINT_RECEIPT.json"
        before = read(receipt_path) if receipt_path.exists() else None
        if before is not None and before["status"] == "COMPLETE":
            return train.fit(fit_id)
        completed = before["completed_updates"] if before else 0
        segment = folder / "RUNTIME_SEGMENTS" / f"{completed:06d}_{time.time_ns()}"
        freeze(
            segment / "START.json",
            {
                "utc": utc(),
                "pid": os.getpid(),
                "fit_id": fit_id,
                "runtime_binding": digest(binding),
                "before_receipt": before,
            },
        )
        result = train.fit(fit_id)
        freeze(
            segment / "END.json",
            {
                "utc": utc(),
                "after_receipt": result,
                "start_sha256": sha(segment / "START.json"),
                "runtime_binding": digest(binding),
            },
        )
        return result

    try:
        factory = partial(MappedSensorDataset, admission=reader)
        # The replacement is explicit, confined to this process, tested by the
        # registered real-P diagnostic, and fully identified in runtime receipts.
        with (
            patch.object(train, "SensorDataset", factory),
            patch.object(primary, "fit", fit_with_provenance),
        ):
            primary.run(once=once)
    finally:
        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    run(once=parser.parse_args().once)
