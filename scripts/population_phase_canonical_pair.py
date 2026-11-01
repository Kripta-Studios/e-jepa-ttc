"""Expand fixed canonical F0, reuse 149 sealed queries and preserve the router failure."""

import argparse
import hashlib
import platform
import shutil
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import configure_cuda
from operational.phase_campaign.data import read
from operational.phase_observation_cache.capture import capture_road
from operational.phase_observation_cache.replay import capture_context
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_submission.runtime import load_submission_model
from operational.phase_timewarp.compact_capture import bind_compact_road, capture_compact
from scripts.build_phase_native_context_augmentations import verify_artifacts
from scripts.final_train40_transfer import FinalTransferPredictor, reject_targets
from scripts.pilot_phase_observation_cache import sensor_row_receipt
from scripts.pilot_phase_producer_timewarp import read_arrays
from scripts.screen_phase_canonical_pair import predict_query

PLAN = Path("docs/superpowers/plans/2026-10-31-phase-canonical-population2608.md")


def completed_query(folder: Path, token: str, group: str, freeze_sha256: str) -> dict:
    """Accept a resumed query only with its complete immutable artifact receipt."""
    complete = read(folder / "COMPLETE.json")
    if (
        complete["token"] != token
        or complete["group"] != group
        or not complete["baseline_exact"]
        or complete["freeze_sha256"] != freeze_sha256
    ):
        raise ValueError("Resumed query binding differs")
    verify_artifacts(folder, complete["artifacts"])
    record = read(folder / "RESULT.json")
    if record["token"] != token or record["group"] != group or not record["baseline_exact"]:
        raise ValueError("Resumed prediction identity differs")
    return record


def main() -> None:
    """Extract only uncached road queries, then seal the whole population before targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in {
        "screen": "output/phase_timewarp/screen149_v1",
        "expanded": "output/phase_observation_cache/expanded149",
        "screen-evaluation": "output/phase_timewarp/screen149_evaluation_v1/REPORT.json",
        "population": "output/phase_submission/training/POPULATION.npz",
        "road-metadata": "output/phase_adapt_final/data_v3/POPULATION.json",
        "config": "output/phase_submission/training/CONFIG.json",
        "bundle": "output/phase_submission/FINAL_MODEL",
        "output": "output/phase_timewarp/population2608_v1",
    }.items():
        parser.add_argument("--" + name, type=Path, default=Path(default))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise ValueError("Use a fresh output or explicitly resume verified complete queries")
    if (args.output / "COMPLETE.json").exists() or (args.output / "FAILED.json").exists():
        raise ValueError("Completed or failed population runs are immutable")
    previous = read(args.screen / "COMPLETE.json")
    screen_freeze = read(args.screen / "FREEZE.json")
    if previous["queries"] != 149 or not previous["baseline_exact"] or previous["targets_read"]:
        raise ValueError("Require sealed identity-verified target-free screen")
    verify_artifacts(args.screen, previous["artifacts"])
    verify_artifacts(Path.cwd(), screen_freeze["source_hashes"])
    decision = read(args.screen_evaluation)
    if (
        decision["screen_complete_sha256"] != file_sha(args.screen / "COMPLETE.json")
        or not decision["routes"]["F0"]["continue_full_population"]
    ):
        raise ValueError("F0 did not pass the fixed screen")
    if file_sha(args.population) != screen_freeze["inputs"][str(args.population)]:
        raise ValueError("Population bytes differ from the fixed screen")
    if (
        file_sha(args.expanded / "COMPLETE.json")
        != screen_freeze["inputs"][str(args.expanded / "COMPLETE.json")]
    ):
        raise ValueError("Original expansion identity differs")
    expanded_complete = read(args.expanded / "COMPLETE.json")
    if file_sha(args.expanded / "FREEZE.json") != expanded_complete["freeze_sha256"]:
        raise ValueError("Original expansion freeze changed")
    expanded_freeze = read(args.expanded / "FREEZE.json")
    for path in (args.road_metadata, args.config):
        if file_sha(path) != expanded_freeze["inputs"][str(path)]:
            raise ValueError("Original sensor metadata or eligibility configuration changed")
    # Read only inference fields. Existing accuracy results justify expansion, never adjustment.
    with np.load(args.population, allow_pickle=False) as stored:
        population = {k: stored[k] for k in ("token", "group", "geometry", "quality")}
    tokens, groups = population["token"].tolist(), population["group"].tolist()
    config = read(args.config)
    if (
        len(tokens) != 2608
        or len(set(tokens)) != 2608
        or groups.count("Slider-500") != 77
        or len(set(groups)) != 25
        or set(groups) & set(config["excluded_road"] + config["excluded_slider"])
    ):
        raise ValueError("Require unchanged 2608-query/25-recording eligible population")
    reuse = {r["token"]: r for r in read(args.screen / "PREDICTIONS.json")["records"]}
    if len(reuse) != 149 or not set(reuse) <= set(tokens):
        raise ValueError("Invalid reused screen cohort")
    metadata = {"evttc:" + r["query_id"]: r for r in read(args.road_metadata)["rows"]}
    paths = verified_upstream_paths(args.bundle / "upstream")
    if read(args.bundle / "upstream/BUNDLE.json")["parent"] != screen_freeze["upstream"]:
        raise ValueError("Frozen upstream model differs")
    sources = set(screen_freeze["source_hashes"]) | {
        str(Path(__file__).resolve().relative_to(Path.cwd())),
        str(PLAN),
        "operational/phase_timewarp/compact_capture.py",
        "tests/unit/test_phase_compact_capture.py",
    }
    hashes = {p: file_sha(Path(p)) for p in sorted(sources)}
    inputs = {
        str(p): file_sha(p)
        for p in (
            args.screen / "COMPLETE.json",
            args.screen_evaluation,
            args.population,
            args.road_metadata,
            args.config,
            args.bundle / "MODEL.json",
        )
    }
    settings = {
        "schema": "phase_canonical_population2608_v1",
        "tokens": tokens,
        "groups": groups,
        "source_hashes": hashes,
        "inputs": inputs,
        "canonical_dt_seconds": 0.1,
        "upstream": screen_freeze["upstream"],
        "device": args.device,
        "seed": 7,
        "threads": 2,
        "tf32": False,
        "targets_read": False,
        "optimizer_updates": 0,
        "primary_route": "F0",
        "router_status": "failed_screen_diagnostic_only",
        "reused_queries": 149,
        "new_road_queries": 2459,
    }
    if args.resume:
        freeze = read(args.output / "FREEZE.json")
        if any(freeze.get(k) != v for k, v in settings.items()):
            raise ValueError("Resume inputs, settings or executed sources differ")
    else:
        freeze = {
            **settings,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "start_time": datetime.now(UTC).isoformat(),
            "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
        }
        atomic_receipt(args.output / "FREEZE.json", freeze)
        for relative in hashes:
            destination = args.output / "executed_source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(relative, destination)
    configure_cuda()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    started, durations = time.perf_counter(), []
    records = []
    try:
        router = load_submission_model(args.bundle)
        runtime = FinalTransferPredictor(paths, args.device)
        for index, (token, group) in enumerate(zip(tokens, groups, strict=True)):
            if token in reuse:
                if reuse[token]["group"] != group:
                    raise ValueError("Reused group differs")
                records.append(reuse[token])
                continue
            if group == "Slider-500":
                raise ValueError("All Slider queries must be reused without new extraction")
            identity = hashlib.sha256(token.encode()).hexdigest()[:24]
            folder = args.output / "queries" / identity
            if (folder / "COMPLETE.json").exists():
                records.append(
                    completed_query(folder, token, group, file_sha(args.output / "FREEZE.json"))
                )
                continue
            if folder.exists():
                raise ValueError(
                    "Incomplete query evidence requires explicit recovery in a fresh run"
                )
            query_start = time.perf_counter()
            row = {k: v for k, v in metadata[token].items() if k != "ttc_csv_path"}
            reject_targets(row)
            atomic_receipt(folder / "QUERY.json", sensor_row_receipt(row))
            with (
                capture_compact(runtime.models) as (calls, arrays),
                capture_context(runtime.models) as context,
            ):
                original, observations, details = capture_road(runtime, row)
            bind_compact_road(
                calls,
                arrays,
                {"event": observations["event"], "rgb": details["rgb_prepared"]},
                observations,
            )
            save_npz(folder / "PAIR_INPUTS.npz", arrays)
            atomic_receipt(folder / "CALLS.json", {"calls": calls})
            atomic_receipt(folder / "EXTRACTION.json", {"result": original, "details": details})
            arrays = read_arrays(folder / "PAIR_INPUTS.npz")
            results = predict_query(
                runtime,
                router,
                calls,
                arrays,
                context,
                population["geometry"][index],
                population["quality"][index],
                row["anchor_us"],
                args.device,
                folder,
            )
            record = {
                "token": token,
                "group": group,
                "query_us": int(row["anchor_us"]),
                "baseline_exact": True,
                "results": results,
            }
            atomic_receipt(folder / "RESULT.json", record)
            atomic_receipt(
                folder / "COMPLETE.json",
                {
                    "token": token,
                    "group": group,
                    "baseline_exact": True,
                    "freeze_sha256": file_sha(args.output / "FREEZE.json"),
                    "artifacts": {
                        str(p.relative_to(folder)): file_sha(p)
                        for p in folder.rglob("*")
                        if p.is_file()
                    },
                },
            )
            records.append(record)
            durations.append(time.perf_counter() - query_start)
            remaining = sum(t not in reuse for t in tokens[index + 1 :])
            eta = float(np.mean(durations[-20:])) * remaining
            print(
                f"{index + 1}/2608 new={len(durations)} {token} {durations[-1]:.2f}s "
                f"elapsed={time.perf_counter() - started:.1f}s rolling_ETA={eta / 60:.1f}min",
                flush=True,
            )
        verify_artifacts(Path.cwd(), hashes)
        for path, digest in inputs.items():
            if file_sha(Path(path)) != digest:
                raise ValueError("Bound input changed during execution")
        atomic_receipt(
            args.output / "PREDICTIONS.json", {"records": records, "targets_read": False}
        )
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "queries": 2608,
                "baseline_exact": True,
                "targets_read": False,
                "optimizer_updates": 0,
                "primary_route": "F0",
                "freeze_sha256": file_sha(args.output / "FREEZE.json"),
                "reused_screen_sha256": file_sha(args.screen / "COMPLETE.json"),
                "elapsed_seconds": time.perf_counter() - started,
                "end_time": datetime.now(UTC).isoformat(),
                "artifacts": {
                    str(p.relative_to(args.output)): file_sha(p)
                    for p in args.output.rglob("*")
                    if p.is_file()
                },
            },
        )
    except Exception as error:
        atomic_receipt(
            args.output / "FAILED.json",
            {"error": repr(error), "traceback": traceback.format_exc(), "targets_read": False},
        )
        raise


if __name__ == "__main__":
    main()
