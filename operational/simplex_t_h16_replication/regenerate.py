"""Reproduce six new head predictions using only an extracted H16 delivery."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = json.loads((root / "CONTENT_MANIFEST.json").read_text(encoding="utf-8"))
    for name, r in manifest["members"].items():
        path = (root / name).resolve(strict=True)
        if (
            not path.is_relative_to(root)
            or path.stat().st_size != r["bytes"]
            or sha(path) != r["sha256"]
        ):
            raise ValueError(f"included member changed: {name}")
    sys.path.insert(0, str(root / "provenance/src"))
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
    from e_jepa_ttc.simplex_t.endpoint import load_endpoint
    from e_jepa_ttc.simplex_t.model import TemporalConfig
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)

    def guard() -> None:
        process = psutil.Process()
        parent = process.parent()
        supervisor = parent if parent and "python" in parent.name().lower() else process
        rss = supervisor.memory_info().rss + sum(
            child.memory_info().rss for child in supervisor.children(recursive=True)
        )
        if psutil.virtual_memory().available < 2 * 1024**3 or rss > 4 * 1024**3:
            raise InterruptedError("registered RAM/RSS limit during regeneration")
        if psutil.disk_usage(str(root)).free - 1024**3 < 10_000_000_000:
            raise InterruptedError("registered disk margin during regeneration")
        if sys.platform == "win32":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (key, ctypes.c_ulonglong)
                    for key in (
                        "total_physical",
                        "available_physical",
                        "total_page_file",
                        "available_page_file",
                        "total_virtual",
                        "available_virtual",
                        "available_extended_virtual",
                    )
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("Windows commitment counter unavailable")
            if status.available_page_file < 1024**3:
                raise InterruptedError("registered commit margin during regeneration")

    guard()
    seal = json.loads((root / "ENDPOINTS.json").read_text(encoding="utf-8"))
    rows = []
    pieces = {}
    for r in seal["fits"]:
        model = load_endpoint(
            root / r["checkpoint"],
            TemporalConfig(**r["model"]),
            seed=r["seed"],
            freeze_sha256=seal["protocol_sha256"],
            train_source_sha256=r["train_source_sha256"],
            endpoint_sha256=r["checkpoint_sha256"],
        )
        frame = pd.read_parquet(root / "publication" / r["key"] / "PREDICTIONS.parquet")
        maximum = 0.0
        for start in range(0, len(frame), 128):
            guard()
            with np.load(
                root / "cached_inputs" / f"H16_fold{r['fold']}_batch{start:05d}.npz",
                allow_pickle=False,
            ) as a:
                inputs = [torch.from_numpy(a[k]) for k in ("features", "times", "valid", "experts")]
            with torch.inference_mode():
                output = model(*inputs)
                prediction = phase_to_ttc(output["point_phase"].to(torch.float64)).numpy()
            f = frame.iloc[start : start + len(prediction)]
            error = float(np.max(np.abs(prediction - f.prediction_ttc_s.to_numpy())))
            maximum = max(maximum, error)
            if error > 1e-10:
                raise ValueError("emitted TTC regeneration exceeds fixed1e-10 tolerance")
            for field in ("raw_location", "raw_residual", "q10", "q90"):
                if not np.array_equal(
                    output[field].numpy().astype(np.float64), f[field].to_numpy()
                ):
                    raise ValueError(f"raw head field differs: {field}")
            for expert in range(3):
                if not np.array_equal(
                    output["relative_cost"].numpy()[:, expert].astype(np.float64),
                    f[f"predicted_relative_cost{expert}"].to_numpy(),
                ):
                    raise ValueError("relative cost field differs")
            losses = 10000 * np.abs(
                benchmark_phase(prediction) - benchmark_phase(f.target_ttc.to_numpy())
            )
            if not np.array_equal(losses, f.loss.to_numpy()):
                raise ValueError("loss regeneration differs")
        pieces.setdefault(r["seed"], []).append(frame)
        rows.append(dict(seed=r["seed"], fold=r["fold"], queries=len(frame), max_ttc_error=maximum))
    scores = {}
    for seed, frames in pieces.items():
        frame = pd.concat(frames, ignore_index=True)
        mass = strict_macro_mass(frame.target_ttc.to_numpy(), frame.sequence_id.to_numpy())
        scores[str(seed)] = float(mass @ frame.loss.to_numpy())
    report = dict(
        status="SIX_NEW_HEADS_REGENERATED_FROM_EXTRACTED_BYTES",
        heads=rows,
        h16_mid=scores,
        members_verified=len(manifest["members"]),
        optimizer_updates=0,
        raw_reconstruction=False,
        torch_version=str(torch.__version__),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
