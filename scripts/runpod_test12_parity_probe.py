"""Plan or replay exactly one frozen public Test12 observation, without training.

Default mode validates provenance and prints a single-file download command.
Execution requires that H5 already exists locally. No downloads, CUDA work,
ground-truth access, global input rewrites, or historical cache overwrites occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import struct
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PureWindowsPath
from typing import Any

INPUTS_SHA = "b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526"
FREEZE_SHA = "b3c4514bebb54112e94ad9ab52f997e5676e3ee89139294ab1b265ee9a650934"
PUBLIC_SHA = "aedb9da5484ceedb642ef1cbc7cf6115dcf7f368e4444b55de947989fc19a362"
MEDIA_REVISION = "a814a946e60d37f7242358af5f344542177e50c5"


def normalize_npz_platform(path: Path) -> int:
    """Preserve compressed members, recording the original Windows ZIP creator.

    NumPy's ZIP writer otherwise emits OS=3 on Linux versus OS=0 on Windows.
    This changes the archive hash despite identical NPY bytes. Only the creator
    OS byte of each validated central-directory record is normalized; the final
    archive still has to match the historical SHA exactly.
    """
    data = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt NPZ before platform normalization")
        entries = archive.infolist()
        cursor = archive.start_dir
    changed = 0
    for entry in entries:
        if data[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError("Unexpected ZIP central-directory layout")
        filename, extra, comment = struct.unpack_from("<HHH", data, cursor + 28)
        if data[cursor + 46 : cursor + 46 + filename].decode("utf-8") != entry.filename:
            raise ValueError("ZIP central-directory member order changed")
        if data[cursor + 5] not in (0, 3):
            raise ValueError("Unrecognized NPZ creator platform")
        changed += int(data[cursor + 5] != 0)
        data[cursor + 5] = 0
        cursor += 46 + filename + extra + comment
    path.write_bytes(data)
    return changed


def digest(path: Path) -> str:
    """Hash a file using bounded memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_bound(path: Path, expected: str) -> dict[str, Any]:
    """Reject changed frozen inputs before interpreting their contents."""
    if digest(path) != expected:
        raise ValueError(f"Frozen identity mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sources(code_root: Path, freeze: dict[str, Any]) -> dict[str, str]:
    """Map historical Windows source paths into one explicitly selected checkout."""
    verified = {}
    for historical, expected in freeze["sources"].items():
        parts = PureWindowsPath(historical).parts
        start = next(i for i, part in enumerate(parts) if part in {"src", "operational"})
        relative = Path(*parts[start:])
        path = code_root / relative
        if digest(path) != expected:
            raise ValueError(f"Preprocessing source differs from frozen version: {path}")
        verified[relative.as_posix()] = expected
    return verified


def prepare_plan(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the first frozen physical observation in one deterministic sequence."""
    inputs = read_bound(args.frozen_inputs, INPUTS_SHA)
    freeze = read_bound(args.event_freeze, FREEZE_SHA)
    if digest(args.public_inputs) != PUBLIC_SHA or freeze["input_sha256"] != PUBLIC_SHA:
        raise ValueError("Public test metadata does not match frozen metadata")
    sources = validate_sources(args.code_root.resolve(), freeze)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8-sig"))
    if inventory["revision"] != MEDIA_REVISION:
        raise ValueError("Media inventory revision changed")
    observations = inputs["event_observations"]
    sequence = args.sequence or min(row["sequence"] for row in freeze["rows"])
    selected = next(
        (
            (number, row)
            for number, row in enumerate(freeze["rows"])
            if row["sequence"] == sequence and row["sample_token"] in observations
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"No frozen physical observation for sequence {sequence}")
    number, row = selected
    observation = observations[row["sample_token"]]
    name = f"query_{number:05d}.npz"
    if PureWindowsPath(observation["current_tensor_path"]).name != name:
        raise ValueError("Frozen row order and tensor filename disagree")
    relative = f"data/test/{sequence}/events.h5"
    pin = next(item for item in inventory["files"] if item["path"] == relative)
    command = [
        "hf",
        "download",
        "NAIL-HNU/eAP-dataset",
        relative,
        "--repo-type",
        "dataset",
        "--revision",
        MEDIA_REVISION,
        "--local-dir",
        str(args.raw_root.resolve()),
    ]
    plan = {
        "schema": "test12_single_query_parity_probe_v1",
        "status": "PLANNED_NOT_EXECUTED",
        "sequence": sequence,
        "sample_token": row["sample_token"],
        "query_index": number,
        "tensor_name": name,
        "expected_tensor_sha256": observation["current_tensor_sha256"],
        "raw_relative_path": relative,
        "raw_bytes": pin["bytes"],
        "raw_sha256": pin["sha256"],
        "media_revision": MEDIA_REVISION,
        "input_freeze_sha256": FREEZE_SHA,
        "inputs_sha256": INPUTS_SHA,
        "public_inputs_sha256": PUBLIC_SHA,
        "inventory_sha256": digest(args.inventory),
        "source_bindings": sources,
        "code_root": str(args.code_root.resolve()),
        "probe_script_sha256": digest(Path(__file__)),
        "download_argv": command,
        "download_command_posix": shlex.join(command),
        "raw_present": (args.raw_root / relative).is_file(),
        "optimizer_updates": 0,
        "test_labels_read": False,
        "scope": "One query only; passing does not certify all Test12 inputs",
    }
    return plan, row


def execute(args: argparse.Namespace, plan: dict[str, Any], row: dict[str, Any]) -> int:
    """Replay the original CPU H8 transform and require exact serialized NPZ SHA."""
    raw = (args.raw_root / plan["raw_relative_path"]).resolve(strict=True)
    raw.relative_to(args.raw_root.resolve())
    if raw.stat().st_size != plan["raw_bytes"]:
        raise ValueError("Raw file size differs from pinned public release")
    print("Verifying one raw H5 SHA256 before preprocessing", flush=True)
    if digest(raw) != plan["raw_sha256"]:
        raise ValueError("Raw file SHA256 differs from pinned public release")
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < 2 * 1024**3:
        raise RuntimeError("Probe requires at least 2 GiB free output space")
    # Import only after every frozen source hash was checked. This executable
    # starts a fresh interpreter; do not import this module into a training worker.
    sys.path[:0] = [str(args.code_root.resolve()), str(args.code_root.resolve() / "src")]
    import numpy as np
    import torch

    from e_jepa_ttc.efficient_context.mapped_union import encode_union
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
    from operational.sota_evidence.test_inputs import supported_job

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    pool = ReaderPool()
    try:
        reader = pool.get(raw)
        stamps = reader.datasets["events/t"]
        relocated = dict(row, path=str(raw))
        job = supported_job(relocated, int(stamps[0]), int(stamps[-1]))
        events = encode_union(
            reader,
            job["windows"],
            np.arange(750000, -1, -50000, dtype=np.int64),
            np.concatenate((np.zeros(8, bool), job["valid"])),
            tuple(job["square"]),
            sequence_id=job["sequence"],
            roi_size=128,
            event_pixel_diff=5,
        ).numpy()[-8:]
        if events.shape != (8, 3, 12, 128, 128) or not np.isfinite(events).all():
            raise ValueError("Prepared H8 tensor violates frozen contract")
        with tempfile.TemporaryDirectory(prefix="single_query_", dir=args.output) as stage:
            candidate = Path(stage) / plan["tensor_name"]
            np.savez_compressed(candidate, events=events, valid=job["valid"])
            native_sha = digest(candidate)
            platform_bytes_changed = normalize_npz_platform(candidate)
            actual = digest(candidate)
            matched = actual == plan["expected_tensor_sha256"]
            destination = args.output / (
                plan["tensor_name"] if matched else f"mismatch_{actual[:16]}.npz"
            )
            if destination.exists() and digest(destination) != actual:
                raise ValueError("Refusing to overwrite different existing probe output")
            os.replace(candidate, destination)
        receipt = dict(
            plan,
            status="EXACT_NPZ_SHA_MATCH" if matched else "PARITY_MISMATCH",
            actual_tensor_sha256=actual,
            native_platform_tensor_sha256=native_sha,
            zip_creator_os_bytes_normalized=platform_bytes_changed,
            actual_tensor_path=str(destination.resolve()),
            events_array_sha256=hashlib.sha256(events.tobytes(order="C")).hexdigest(),
            valid=job["valid"].tolist(),
            preprocessing_seconds=time.monotonic() - started,
            python=platform.python_version(),
            numpy=np.__version__,
            torch=torch.__version__,
        )
        receipt_path = args.output / f"{plan['tensor_name']}.parity.json"
        temporary = receipt_path.with_suffix(".pending.json")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, receipt_path)
        print(json.dumps(receipt, indent=2))
        return 0 if matched else 2
    finally:
        pool.close()


def main() -> int:
    """Validate a one-query plan, or execute only with an explicit --execute flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "frozen-inputs",
        "event-freeze",
        "public-inputs",
        "inventory",
        "code-root",
        "raw-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sequence", help="Default: lexicographically first frozen sequence")
    parser.add_argument("--execute", action="store_true", help="Replay one query from existing H5")
    args = parser.parse_args()
    plan, row = prepare_plan(args)
    if args.execute:
        return execute(args, plan, row)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
