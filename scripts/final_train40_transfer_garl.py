"""Frozen full Garl baseline on a separately registered, sensor-only EvTTC cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    """Hash bytes without materializing a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic(path: Path, value: dict[str, Any]) -> None:
    """Durably publish complete JSON, never a partially written fragment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def guard(minimum: int) -> None:
    import psutil

    available = psutil.virtual_memory().available
    if available < minimum:
        raise MemoryError(f"RAM guard: {available} available bytes < {minimum}")


def validate_fragment(
    value: dict[str, Any], row: dict[str, Any], manifest: str, binding: str
) -> None:
    """Reject foreign, malformed, or scored fragments on resume."""
    if (
        value.get("query_id") != row["query_id"]
        or value.get("manifest_sha256") != manifest
        or value.get("binding_sha256") != binding
        or value.get("targets_read") is not False
        or value.get("optimizer_updates") != 0
    ):
        raise ValueError("Saved Garl fragment identity changed")
    prediction = value.get("prediction_ttc")
    if prediction is not None and (
        isinstance(prediction, bool)
        or not isinstance(prediction, (int, float))
        or not math.isfinite(prediction)
    ):
        raise ValueError("Invalid scalar prediction")
    if value.get("status") == "PREDICTED" and prediction is None:
        raise ValueError("Missing predicted TTC")
    if value.get("status") not in {"PREDICTED", "NONFINITE_NATIVE_TTC", "EMPTY_EVENT_SEGMENT"}:
        raise ValueError("Unknown Garl status")


def execute(args: argparse.Namespace) -> None:
    import psutil

    torch = importlib.import_module("torch")

    from operational.evttc_rgb_transfer.model import (
        _NATIVE_SOURCES,
        CHECKPOINT_SHA256,
        CONFIG_SHA256,
        HF_REVISION,
        FullGarl,
    )
    from scripts.post_v18_transfer_inputs import garl_inputs

    manifest_sha, registration_sha = sha(args.manifest), sha(args.registration)
    manifest, registration = read(args.manifest), read(args.registration)
    if manifest.get("targets_read") is not False or registration.get("targets_read") is not False:
        raise ValueError("Sensor-only cohort declaration required")
    if registration.get("manifest_sha256") != manifest_sha:
        raise ValueError("Cohort registration does not bind this manifest")
    rows = manifest["rows"]
    if not rows or len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate query population")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source_paths = [
        Path(__file__),
        ROOT / "scripts/post_v18_transfer_inputs.py",
        ROOT / "operational/evttc_rgb_transfer/model.py",
        ROOT / "operational/evttc_rgb_transfer/inputs.py",
        ROOT / "operational/evttc_transfer/inputs.py",
        ROOT / "src/e_jepa_ttc/data/garl_official_preprocessing.py",
        ROOT / "src/e_jepa_ttc/efficient_context/garl_input.py",
    ]
    if sha(args.public_dir / "paper_ours_full.pth") != CHECKPOINT_SHA256:
        raise ValueError("Official checkpoint hash differs")
    native_sources = {name: sha(args.code_root / name) for name in _NATIVE_SOURCES}
    if native_sources != _NATIVE_SOURCES:
        raise ValueError("Native Garl source drift")
    binding = {
        "schema": "final_train40_transfer_garl_binding_v1",
        "model_name": "Garl_full",
        "checkpoint_path": str((args.public_dir / "paper_ours_full.pth").resolve()),
        "native_source_sha256": native_sources,
        "modality": "RGB+events",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "hf_revision": HF_REVISION,
        "manifest_sha256": manifest_sha,
        "registration_sha256": registration_sha,
        "manifest_path": str(args.manifest.resolve()),
        "registration_path": str(args.registration.resolve()),
        "public_dir": str(args.public_dir.resolve()),
        "native_code_root": str(args.code_root.resolve()),
        "device": args.device,
        "precision": "float32",
        "threads": args.threads,
        "sources": {str(path.resolve()): sha(path) for path in source_paths},
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "numpy", "h5py", "psutil", "PyYAML"]
        },
        "python": platform.python_version(),
        "host": platform.node(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "native_delta_t_s": 0.1,
        "clipping": None,
        "targets_read": False,
        "optimizer_updates": 0,
    }
    freeze = args.output / "INFERENCE_FREEZE.json"
    if freeze.exists():
        if read(freeze) != binding:
            raise ValueError("Garl frozen run binding differs; use a new output directory")
    else:
        atomic(freeze, binding)
    binding_sha = sha(freeze)
    pending = []
    for index, row in enumerate(rows):
        path = args.output / f"query_{index:06d}.json"
        if path.exists():
            validate_fragment(read(path), row, manifest_sha, binding_sha)
        else:
            pending.append((index, row, path))
    started = time.monotonic()
    model = None
    if pending:
        guard(args.min_ram_bytes)
        model = FullGarl(args.public_dir, args.code_root, args.device)
        atomic(args.output / "MODEL.json", model.bindings)
    for ordinal, (index, row, path) in enumerate(pending):
        if args.limit is not None and ordinal >= args.limit:
            break
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("Garl wall budget reached; complete fragments retained")
        if (args.output / "STOP_REQUEST").exists():
            atomic(
                args.output / "STATUS.json", {"status": "STOPPED_MANUAL", "timestamp": time.time()}
            )
            return
        guard(args.min_ram_bytes)
        start = time.perf_counter()
        prepared = garl_inputs(row)
        middle = time.perf_counter()
        assert model is not None
        prediction = (
            model.predict_sensor(prepared["sensor"]) if prepared["sensor"] is not None else None
        )
        end = time.perf_counter()
        value = None if prediction is None else prediction["ttc"]
        if value is not None and not isinstance(value, float):
            raise TypeError("Expected native TTC float")
        status = (
            prepared["status"]
            if prediction is None
            else (
                "PREDICTED"
                if value is not None and math.isfinite(value)
                else "NONFINITE_NATIVE_TTC"
            )
        )
        result = {
            "query_id": row["query_id"],
            "manifest_sha256": manifest_sha,
            "binding_sha256": binding_sha,
            "status": status,
            "failure_reason": None if status == "PREDICTED" else status,
            "prediction_ttc": value if value is not None and math.isfinite(value) else None,
            "heights": None if prediction is None else prediction["heights"],
            "sensor_sha256": prepared.get("sensor_sha256"),
            "seconds": {"preprocess": middle - start, "model": end - middle, "total": end - start},
            "timestamp": time.time(),
            "pid": os.getpid(),
            "rss_bytes": psutil.Process().memory_info().rss,
            "targets_read": False,
            "optimizer_updates": 0,
        }
        validate_fragment(result, row, manifest_sha, binding_sha)
        atomic(path, result)
        progress = {
            "status": "RUNNING",
            "index": index,
            "query_id": row["query_id"],
            "timestamp": time.time(),
            "seconds": end - start,
            "pid": os.getpid(),
        }
        atomic(args.output / "STATUS.json", progress)
        print(json.dumps(progress), flush=True)
    fragments = {}
    for index, row in enumerate(rows):
        path = args.output / f"query_{index:06d}.json"
        if not path.exists():
            return
        validate_fragment(read(path), row, manifest_sha, binding_sha)
        fragments[path.name] = sha(path)
    atomic(
        args.output / "PREDICTIONS_SEALED.json",
        {
            "schema": "final_train40_transfer_predictions_v1",
            "status": "COMPLETE",
            "manifest_sha256": manifest_sha,
            "binding_sha256": binding_sha,
            "queries": len(rows),
            "fragments": fragments,
            "targets_read": False,
            "optimizer_updates": 0,
        },
    )
    atomic(
        args.output / "STATUS.json",
        {"status": "COMPLETE", "queries": len(rows), "timestamp": time.time()},
    )


def main() -> None:
    """CLI: no GT reads, no training, fixed scientific preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "artifacts/final_train40_transfer_20261027"
    parser.add_argument("--manifest", type=Path, default=base / "QUERY_MANIFEST.json")
    parser.add_argument("--registration", type=Path, default=base / "COHORT_REGISTRATION.json")
    parser.add_argument("--output", type=Path, default=base / "garl")
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=ROOT.parent
        / "e-jepa-ttc-v12-efficient-context/artifacts/evttc_rgb_transfer_20261008/public_garl",
    )
    parser.add_argument("--code-root", type=Path, default=Path("E:/Garl-TTC"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-ram-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--max-wall-seconds", type=float, default=7200)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.min_ram_bytes < 2 * 1024**3 or args.threads < 1 or args.max_wall_seconds <= 0:
        parser.error("Invalid resource bounds")
    try:
        from filelock import FileLock

        args.output.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.output / "WORKER.lock"), timeout=0):
            execute(args)
    except Exception as error:
        atomic(
            args.output / "STATUS.json",
            {
                "status": "FAILED",
                "type": type(error).__name__,
                "error": str(error),
                "timestamp": time.time(),
                "pid": os.getpid(),
            },
        )
        raise


if __name__ == "__main__":
    main()
