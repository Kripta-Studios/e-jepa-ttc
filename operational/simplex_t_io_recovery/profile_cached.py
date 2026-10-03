"""Repeat registered head timing from target-free inputs in an extracted delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path


def sha(path: Path) -> str:
    """Hash an included numerical payload with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: dict) -> None:
    """Keep each confirmed profiling fragment durable and recoverable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".next")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def main() -> None:
    """Measure CPU FP32 heads and emission; no experts, raw data, targets or optimizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = json.loads((root / "CONTENT_MANIFEST.json").read_bytes())
    index_path = root / "PROFILE_INPUT_EXPORT.json"
    if sha(index_path) != manifest["members"][index_path.name]["sha256"]:
        raise ValueError("profiling export index changed")
    index = json.loads(index_path.read_bytes())
    if index["status"] != "NINE_PREPARED_HEAD_PROFILE_INPUTS_EXPORTED" or len(index["models"]) != 9:
        raise ValueError("complete target-free nine-model profile export required")
    guard_source = root / "regenerate.py"
    if sha(guard_source) != manifest["members"][guard_source.name]["sha256"]:
        raise ValueError("canonical profiling resource guard changed")
    guards = runpy.run_path(str(guard_source), run_name="cached_profile_guard")
    check = guards["resource_guard"]
    check(root)
    sys.path[:0] = [str(root / "provenance"), str(root / "provenance/src")]
    import numpy as np
    import torch

    from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from operational.simplex_t_cost_context.model import build_model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    reports = []
    for label, row in index["models"].items():
        check(root)
        paths = []
        for kind in ("weights", "inputs"):
            path = (root / row[kind + "_path"]).resolve(strict=True)
            if not path.is_relative_to(root) or sha(path) != row[kind + "_sha256"]:
                raise ValueError("profiling weights or inputs changed")
            paths.append(path)
        constructor = row["constructor"]
        model = (
            TemporalRefiner(TemporalConfig(**constructor["config"]))
            if constructor["kind"] == "historical_temporal"
            else build_model(constructor["arm"])
        )
        model.float().cpu().eval()
        with np.load(paths[0], allow_pickle=False) as archive:
            model.load_state_dict(
                {key: torch.from_numpy(archive[key].copy()) for key in archive.files}, strict=True
            )
        with np.load(paths[1], allow_pickle=False) as archive:
            if set(archive.files) != {"features", "times", "valid", "experts"}:
                raise ValueError("only target-free prepared inputs are accepted")
            inputs = tuple(
                torch.from_numpy(archive[key].copy())
                for key in ("features", "times", "valid", "experts")
            )
        timing, throughput = [], []
        with torch.inference_mode():
            for i in range(525):
                check(root)
                position = (i if i < 25 else i - 25) % 64
                batch = tuple(value[position : position + 1] for value in inputs)
                start = time.perf_counter_ns()
                output = model(*batch)
                phase_to_ttc(output["point_phase"].to(torch.float64))
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if not all(bool(torch.isfinite(value).all()) for value in output.values()):
                    raise ValueError("nonfinite cached profiling output")
                if i >= 25:
                    timing.append(elapsed)
                    if len(timing) % 25 == 0:
                        save(
                            args.output.with_suffix(".progress.json"),
                            dict(
                                status="CACHED_PROFILE_FRAGMENT_COMMITTED",
                                model=label,
                                measurements=len(timing),
                                raw_ms=timing,
                                optimizer_updates=0,
                                selection_sha256=row["selection_sha256"],
                            ),
                        )
            batch = tuple(value.repeat((2,) + (1,) * (value.ndim - 1)) for value in inputs)
            for _ in range(50):
                check(root)
                start = time.perf_counter_ns()
                phase_to_ttc(model(*batch)["point_phase"].to(torch.float64))
                throughput.append((time.perf_counter_ns() - start) / 1e6)
        reports.append(
            dict(
                model=label,
                raw_ms=timing,
                p50_ms=float(np.median(timing)),
                p95_ms=float(np.quantile(timing, 0.95)),
                batch=1,
                measurements=500,
                batch128_windows_per_second=128000 / float(np.median(throughput)),
                selection_sha256=row["selection_sha256"],
                scope="prepared_input_head_only",
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    status="CACHED_HEAD_PROFILE_FRAGMENT_COMMITTED",
                    models=reports,
                    optimizer_updates=0,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        del model
    args.output.write_text(
        json.dumps(
            dict(
                status="NINE_CACHED_HEAD_PROFILES_COMPLETE",
                models=reports,
                optimizer_updates=0,
                raw_or_expert_cost_measured=False,
                timing_is_host_dependent=True,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
