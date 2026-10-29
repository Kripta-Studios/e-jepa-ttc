"""Verify new C0 heads from extracted cached inputs; no training or raw expert calls."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def sha(path: Path) -> str:
    """Hash complete member bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path) -> dict:
    """Read one explicit delivery receipt."""
    return json.loads(path.read_text(encoding="utf-8"))


def resource_guard(root: Path) -> dict:
    """Check portable delivery-tree limits without importing the original worktree."""
    import psutil

    process = psutil.Process()
    owner = process
    # Include the delivery parent and lightweight Python driver when present.
    # Never enumerate or alter unrelated trainer/application processes.
    while (parent := owner.parent()) is not None:
        if "python" not in parent.name().lower():
            break
        owner = parent
    rss = 0
    for member in [owner, *owner.children(recursive=True)]:
        try:
            rss += member.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    available = int(psutil.virtual_memory().available)
    free_after_reserve = shutil.disk_usage(root).free - 1024**3
    commit_headroom = None
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_pagefile", ctypes.c_ulonglong),
                ("available_pagefile", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel.GlobalMemoryStatusEx
        query.argtypes = [ctypes.POINTER(MemoryStatus)]
        query.restype = ctypes.c_int
        if not query(ctypes.byref(memory)):
            raise ctypes.WinError(ctypes.get_last_error())
        commit_headroom = int(memory.available_pagefile)
    observed = dict(
        available_ram_bytes=available,
        delivery_tree_rss_bytes=rss,
        tree_owner_pid=owner.pid,
        disk_free_after_reserve_bytes=free_after_reserve,
        disk_reserve_bytes=1024**3,
        windows_commit_headroom_bytes=commit_headroom,
    )
    failures = []
    if available < 2 * 1024**3:
        failures.append("available RAM below 2 GiB")
    if rss > 4 * 1024**3:
        failures.append("delivery Python tree RSS above 4 GiB")
    if free_after_reserve < 10_000_000_000:
        failures.append("disk free after 1 GiB reserve below 10 GB")
    if commit_headroom is not None and commit_headroom < 1024**3:
        failures.append("Windows commit headroom below 1 GiB")
    if failures:
        raise InterruptedError("; ".join(failures) + ": " + json.dumps(observed))
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execution", default="execution")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    resource_guard(root)
    manifest = record(root / "CONTENT_MANIFEST.json")
    for name, row in manifest["members"].items():
        path = (root / name).resolve(strict=True)
        if (
            not path.is_relative_to(root)
            or path.stat().st_size != row["bytes"]
            or sha(path) != row["sha256"]
        ):
            raise ValueError(f"included member differs: {name}")
    sys.path[:0] = [str(root / "provenance"), str(root / "provenance/src")]
    import numpy as np
    import pandas as pd
    import torch

    from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from operational.simplex_t_cost_context.model import build_model

    torch.set_num_threads(4)
    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    execution = (root / args.execution).resolve(strict=True)
    if not execution.is_relative_to(root):
        raise ValueError("execution payload must be inside extracted root")
    index = record(execution / "ANALYSIS_EXPORT_INDEX.json")
    reports, pieces = [], {}
    for key, row in sorted(index["fits"].items()):
        resource_guard(root)
        model = build_model(row["arm"]).float().cpu()
        weight = execution / row["weights_path"]
        if sha(weight) != row["weights_sha256"]:
            raise ValueError("compact model weights differ")
        with np.load(weight, allow_pickle=False) as archive:
            model.load_state_dict(
                {name: torch.from_numpy(archive[name].copy()) for name in archive.files},
                strict=True,
            )
        model.eval()
        prediction_path = execution / row["predictions_path"]
        if sha(prediction_path) != row["predictions_sha256"]:
            raise ValueError("prediction table differs")
        frame = pd.read_parquet(prediction_path)
        maximum = 0.0
        regenerated_losses = []
        for fragment in row["fragments"]:
            resource_guard(root)
            input_path = execution / fragment["input_path"]
            if sha(input_path) != fragment["input_sha256"]:
                raise ValueError("cached fragment inputs differ")
            receipt = execution / fragment["receipt_path"]
            if sha(receipt) != fragment["receipt_sha256"]:
                raise ValueError("fragment receipt differs")
            with np.load(input_path, allow_pickle=False) as archive:
                inputs = [
                    torch.from_numpy(archive[name].copy())
                    for name in ("features", "times", "valid", "experts")
                ]
            with torch.inference_mode():
                output = model(*inputs)
                prediction = phase_to_ttc(output["point_phase"].to(torch.float64)).numpy()
            start, stop = fragment["start"], fragment["stop"]
            observed = frame.iloc[start:stop]
            if len(prediction) != len(observed):
                raise ValueError("fragment query count differs")
            error = float(np.max(np.abs(prediction - observed.prediction_ttc_s.to_numpy())))
            maximum = max(maximum, error)
            if error > 1e-10:
                raise ValueError("emitted TTC differs beyond fixed tolerance1e-10")
            for field in ("raw_location", "raw_residual", "q10", "q90"):
                if not np.array_equal(
                    output[field].numpy().astype(np.float64),
                    observed[field].to_numpy().astype(np.float64),
                ):
                    raise ValueError(f"raw model field differs: {field}")
            for expert in range(3):
                if not np.array_equal(
                    output["relative_cost"].numpy()[:, expert].astype(np.float64),
                    observed[f"predicted_relative_cost{expert}"].to_numpy(),
                ):
                    raise ValueError("inactive cost output differs")
            losses = 10000 * np.abs(
                benchmark_phase(prediction) - benchmark_phase(observed.target_ttc.to_numpy())
            )
            if not np.array_equal(losses, observed.loss.to_numpy()):
                raise ValueError("recomputed emitted-TTC loss differs")
            regenerated_losses.extend(losses.tolist())
        if len(regenerated_losses) != len(frame):
            raise ValueError("fragment coverage differs")
        pieces.setdefault(row["arm"], []).append(frame)
        reports.append(
            dict(
                key=key,
                arm=row["arm"],
                fold=row["fold"],
                queries=len(frame),
                max_ttc_error=maximum,
                fragments=len(row["fragments"]),
            )
        )
        del model
    scores = {}
    for arm, parts in pieces.items():
        frame = pd.concat(parts, ignore_index=True)
        if len(parts) != 3 or len(frame) != 8192 or frame.sample_token.duplicated().any():
            raise ValueError("only complete three-fold arm exports accepted")
        mass = strict_macro_mass(frame.target_ttc.to_numpy(), frame.sequence_id.to_numpy())
        scores[arm] = float(mass @ frame.loss.to_numpy())
    report = dict(
        status="INCLUDED_NEW_HEADS_REGENERATED_FROM_EXTRACTED_CACHED_BYTES",
        heads=reports,
        scores=scores,
        members_verified=len(manifest["members"]),
        optimizer_updates=0,
        raw_reconstruction=False,
        expert_weights_required=False,
        torch_version=str(torch.__version__),
        resource_scope=(
            "Delivery Python tree checked before each head and fragment; "
            "external supervisor retains union budget"
        ),
        last_resource_check=resource_guard(root),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
