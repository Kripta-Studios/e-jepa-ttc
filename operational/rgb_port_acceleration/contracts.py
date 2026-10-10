"""Immutable contracts for the narrowly admitted E_A5 CUDA-graph runtime."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from operational.rgb_port.accounting import (
    atomic_write_bytes,
    atomic_write_json,
    read_bytes_shared,
    read_json_shared,
    sha256_file,
)
from operational.rgb_port_revision.migration import source_matches

FREEZE_SCHEMA = "rgb_port_acceleration_freeze_v1"
ADMISSION_SCHEMA = "rgb_port_e_a5_cudagraph_admission_v1"
FIT_ID = "E_A5_MATCHED"
ORIGIN_UPDATE = 6189
SHAPE = {"batch_size": 32, "frames": 3}
EXCLUDED_ROUTES = ["E_C2F_MATCHED", "R_A5", "R_C2F", "PAIR", "HEADS", "FUSION"]
METRICS_FIT_IDS = ["E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F"]


def _canonical_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != "identity_sha256"}


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_paths(repository: Path) -> list[Path]:
    paths = [
        repository / "operational/rgb_port_acceleration/producer.py",
        repository / "operational/rgb_port/train_producers.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Acceleration source is incomplete: {missing}")
    return paths


def _orchestration_paths(repository: Path) -> list[Path]:
    paths = [
        repository / "operational/rgb_port_acceleration/__init__.py",
        repository / "operational/rgb_port_acceleration/contracts.py",
        repository / "operational/rgb_port_acceleration/queue.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Acceleration orchestration source is incomplete: {missing}")
    return paths


def _torch_backend_paths() -> list[Path]:
    import torch._dynamo.backends.cudagraphs
    import torch.cuda.graphs

    result: list[Path] = []
    for module in (torch.cuda.graphs, torch._dynamo.backends.cudagraphs):
        source = inspect.getsourcefile(module)
        if source is None:
            raise RuntimeError(f"Cannot bind Torch backend source for {module.__name__}")
        result.append(Path(source).resolve(strict=True))
    return result


def build_acceleration_freeze(
    run: Path, admission: Path, output: Path | None = None
) -> dict[str, Any]:
    """Build the supplemental freeze without changing the V2 scientific identity."""
    run = run.resolve(strict=True)
    repository = Path(__file__).resolve().parents[2]
    admission = admission.resolve(strict=True)
    admission_value = read_json_shared(admission)
    if (
        admission_value.get("schema") != ADMISSION_SCHEMA
        or admission_value.get("status") != "PASSED"
        or admission_value.get("fit_id") != FIT_ID
        or admission_value.get("backend") != "cudagraphs"
        or admission_value.get("completed_updates") != ORIGIN_UPDATE
        or admission_value.get("next_cursor", {}).get("shape") != [32, 3]
        or not admission_value.get("compatible")
        or not admission_value.get("checkpoint_unchanged")
        or admission_value.get("optimizer_updates") != 0
    ):
        raise ValueError("CUDA-graph admission is outside the authorized E_A5 B32xT3 scope")
    original_freeze = run / "SOURCE_FREEZE.json"
    fit = run / "fits" / FIT_ID
    receipt = read_json_shared(fit / "CHECKPOINT_RECEIPT.json")
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    checkpoint = fit / "checkpoint_versions" / str(pointer.get("version", ""))
    if (
        receipt.get("completed_updates") != ORIGIN_UPDATE
        or pointer.get("completed_updates") != ORIGIN_UPDATE
        or receipt.get("identity_sha256") != pointer.get("identity_sha256")
        or receipt.get("checkpoint_sha256") != pointer.get("checkpoint_sha256")
        or not checkpoint.is_file()
        or sha256_file(checkpoint) != pointer.get("checkpoint_sha256")
    ):
        raise ValueError("E_A5 native origin is not the sealed update-6189 full checkpoint")
    bindings = admission_value.get("bindings")
    if not isinstance(bindings, dict) or (
        bindings.get("source_freeze_sha256") != sha256_file(original_freeze)
        or bindings.get("checkpoint_sha256") != pointer.get("checkpoint_sha256")
        or bindings.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
        or bindings.get("train_producers_sha256")
        != sha256_file(repository / "operational/rgb_port/train_producers.py")
        or bindings.get("torch_version") != str(torch.__version__)
        or bindings.get("cuda_version") != torch.version.cuda
    ):
        raise ValueError("CUDA-graph admission bindings differ from the frozen runtime origin")
    origin_copy = run / "acceleration" / "E_A5_ORIGIN_006189.pt"
    if origin_copy.exists() and sha256_file(origin_copy) != pointer["checkpoint_sha256"]:
        raise RuntimeError("Immutable acceleration origin copy changed")
    if not origin_copy.exists():
        atomic_write_bytes(origin_copy, read_bytes_shared(checkpoint))
    if sha256_file(origin_copy) != pointer["checkpoint_sha256"]:
        raise RuntimeError("Immutable acceleration origin copy was not byte exact")
    admission_copy = run / "acceleration" / "E_A5_CUDAGRAPH_ADMISSION.json"
    if admission_copy.exists() and read_json_shared(admission_copy) != admission_value:
        raise RuntimeError("Bundled acceleration admission changed")
    if not admission_copy.exists():
        atomic_write_json(admission_copy, admission_value)
    payload: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN",
        "fit_id": FIT_ID,
        "graph_fit_id": FIT_ID,
        "metrics_fit_ids": METRICS_FIT_IDS,
        "runtime_only": True,
        "scientific_identity_unchanged": True,
        "original_source_freeze_path": str(original_freeze.resolve(strict=True)),
        "original_source_freeze_sha256": sha256_file(original_freeze),
        "origin": {
            "completed_updates": ORIGIN_UPDATE,
            "checkpoint_path": str(origin_copy.resolve(strict=True)),
            "checkpoint_sha256": sha256_file(origin_copy),
            "identity_sha256": receipt.get("identity_sha256"),
        },
        "admission_path": str(admission_copy.resolve()),
        "admission_sha256": sha256_file(admission_copy),
        "admission_source_sha256": sha256_file(admission),
        "backend": "cudagraphs",
        "admitted_shape": SHAPE,
        "excluded_routes": EXCLUDED_ROUTES,
        "source_sha256": {
            str(path.resolve()): sha256_file(path) for path in _source_paths(repository)
        },
        "orchestration_source_sha256": {
            str(path.resolve()): sha256_file(path) for path in _orchestration_paths(repository)
        },
        "torch_backend_source_sha256": {
            str(path): sha256_file(path) for path in _torch_backend_paths()
        },
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    payload["identity_sha256"] = canonical_sha256(_canonical_payload(payload))
    target = output.resolve() if output is not None else run / "ACCELERATION_FREEZE.json"
    if target.exists() and read_json_shared(target) != payload:
        raise RuntimeError("Supplemental acceleration freeze changed")
    if not target.exists():
        atomic_write_json(target, payload)
    return payload


def validate_acceleration_freeze(path: Path) -> dict[str, Any]:
    """Fail closed if any sidecar, admission, backend, V2, or origin binding changed."""
    value = read_json_shared(path)
    if (
        value.get("schema") != FREEZE_SCHEMA
        or value.get("status") != "FROZEN"
        or value.get("fit_id") != FIT_ID
        or value.get("graph_fit_id") != FIT_ID
        or value.get("metrics_fit_ids") != METRICS_FIT_IDS
        or value.get("backend") != "cudagraphs"
        or value.get("admitted_shape") != SHAPE
        or value.get("excluded_routes") != EXCLUDED_ROUTES
        or value.get("identity_sha256") != canonical_sha256(_canonical_payload(value))
    ):
        raise ValueError("Acceleration freeze contract differs")
    for field in (
        "source_sha256",
        "orchestration_source_sha256",
        "torch_backend_source_sha256",
    ):
        entries = value.get(field)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Acceleration freeze lacks {field}")
        for source, digest in entries.items():
            if not source_matches(Path(source), digest, path.parent):
                raise RuntimeError(f"Frozen acceleration source changed: {source}")
    if (
        sha256_file(Path(value["original_source_freeze_path"]))
        != value["original_source_freeze_sha256"]
    ):
        raise RuntimeError("Original V2 SOURCE_FREEZE changed")
    if sha256_file(Path(value["admission_path"])) != value["admission_sha256"]:
        raise RuntimeError("Acceleration admission changed")
    origin = value["origin"]
    if origin.get("completed_updates") != ORIGIN_UPDATE or sha256_file(
        Path(origin["checkpoint_path"])
    ) != origin.get("checkpoint_sha256"):
        raise RuntimeError("Native update-6189 acceleration origin changed")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--admission", type=Path, required=True)
    build.add_argument("--output", type=Path)
    check = sub.add_parser("validate")
    check.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    value = (
        build_acceleration_freeze(args.run, args.admission, args.output)
        if args.action == "build"
        else validate_acceleration_freeze(args.freeze)
    )
    print(value["identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
