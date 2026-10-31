"""Replay a frozen two-query sensor cache with one explicit PAIR clock intervention."""

import argparse
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import configure_cuda
from operational.phase_campaign.data import read
from operational.phase_observation_cache.replay import replay
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_timewarp.canonical_pair import CanonicalPair, CanonicalPhaseDomainError
from operational.phase_timewarp.capture import PAIRS
from operational.phase_timewarp.pilot import compare_stages, finite_ratio, replay_calls
from operational.rgb_port import transfer as engine
from scripts.build_phase_native_context_augmentations import native_vector, verify_artifacts
from scripts.final_train40_transfer import FinalTransferPredictor
from scripts.pilot_phase_producer_timewarp import exact_arrays, read_arrays, save_execution


def execute(
    runtime: FinalTransferPredictor,
    calls: list,
    arrays: dict,
    query_us: int,
    speed: float,
    device: str,
    destination: Path,
    only_ids: set[str] | None = None,
) -> dict:
    """Capture corrected PAIR internals alongside the unchanged downstream execution."""
    trace_arrays = {}
    handles = []
    active = {"call_id": -1}
    domain_errors: dict[int, str] = {}
    original_expert = engine._expert_block

    def counted_expert(
        models: dict, prefix: str, pixels: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active["call_id"] += 1
        try:
            return original_expert(models, prefix, pixels, delta)
        except CanonicalPhaseDomainError as error:
            domain_errors[active["call_id"]] = str(error)
            raise ValueError("TTC lies outside the signed benchmark-phase domain") from error

    def hook(name: str) -> Callable:
        def observe(module: CanonicalPair, inputs: tuple, output: torch.Tensor | None) -> None:
            key = f"call{active['call_id']}.{name}"
            trace_arrays[key + ".input133"] = inputs[0].features.detach().cpu().numpy().copy()
            trace_arrays[key + ".returned"] = np.array(output is not None)
            for field, value in module.last_trace.items():
                trace_arrays[key + "." + field] = value.detach().cpu().numpy().copy()

        return observe

    for name in PAIRS:
        handles.append(runtime.models[name].register_forward_hook(hook(name), always_call=True))
    try:
        with patch.object(engine, "_expert_block", counted_expert):
            result = replay_calls(
                runtime.models, calls, arrays, query_us, speed, device, only_ids=only_ids
            )
    finally:
        for handle in handles:
            handle.remove()
    if active["call_id"] + 1 != len(result["calls"]):
        raise ValueError("Canonical trace/call alignment differs")
    for record in result["calls"]:
        if record["call_id"] in domain_errors:
            record["canonical_domain_error"] = domain_errors[record["call_id"]]
    save_npz(destination / "CANONICAL_TRACES.npz", trace_arrays)
    ttc, context = save_execution(destination, result, runtime, query_us, device)
    result.update(ttc=ttc, native=native_vector(context, device))
    return result


def main() -> None:
    """Freeze and run the preregistered no-label, no-update timing intervention."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/phase_timewarp/pilot_v2"))
    parser.add_argument("--output", type=Path, default=Path("output/phase_timewarp/canonical_v1"))
    parser.add_argument(
        "--upstream", type=Path, default=Path("output/phase_submission/FINAL_MODEL/upstream")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output; execution evidence is immutable")
    complete = read(args.source / "COMPLETE.json")
    if (
        complete["status"] != "COMPLETE"
        or complete["queries"] != 2
        or not complete["identity_exact"]
        or complete["targets_read"]
        or complete["optimizer_updates"] != 0
    ):
        raise ValueError("Require completed identity-verified two-query no-label source")
    verify_artifacts(args.source, complete["artifacts"])
    old_freeze = read(args.source / "FREEZE.json")
    if file_sha(args.source / "FREEZE.json") != complete["freeze_sha256"]:
        raise ValueError("Source freeze changed")
    paths = verified_upstream_paths(args.upstream)
    if read(args.upstream / "BUNDLE.json")["parent"] != old_freeze["upstream"]:
        raise ValueError("Upstream model changed")
    for relative, digest in old_freeze["source_hashes"].items():
        if file_sha(Path(relative)) != digest:
            raise ValueError(f"Previous execution source changed: {relative}")
    sources = set(map(Path, old_freeze["source_hashes"])) | {
        Path(__file__),
        Path("operational/phase_timewarp/canonical_pair.py"),
        Path("tests/unit/test_phase_canonical_pair.py"),
        Path("tests/unit/test_phase_canonical_pilot.py"),
        Path("docs/superpowers/plans/2026-10-31-phase-canonical-pair-pilot.md"),
    }
    hashes = {str(p.resolve().relative_to(Path.cwd())): file_sha(p) for p in sources}
    freeze = {
        "source_complete_sha256": file_sha(args.source / "COMPLETE.json"),
        "source_hashes": hashes,
        "upstream": old_freeze["upstream"],
        "canonical_dt_seconds": 0.1,
        "targets_read": False,
        "optimizer_updates": 0,
        "intervention": (
            "Raw PAIR phase canonical-clock inverse-TTC scaling; outer emission unchanged"
        ),
        "device": args.device,
        "torch": torch.__version__,
        "seed": 7,
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
    start = time.perf_counter()
    try:
        runtime = FinalTransferPredictor(paths, args.device)
        originals = dict(runtime.models)
        corrected = {**originals, **{name: CanonicalPair(originals[name]) for name in PAIRS}}
        reports = {}
        for name in ("road", "slider500"):
            source = args.source / name
            calls = read(source / "CAPTURE_CALLS.json")["calls"]
            arrays = read_arrays(source / "CAPTURE_ARRAYS.npz")
            query_us = read(source / "CAPTURE_SEAL.json")["query_us"]
            runtime.models = originals
            baseline = replay_calls(originals, calls, arrays, query_us, 1.0, args.device)
            exact_arrays(baseline["arrays"], arrays, "cached frozen baseline producers")
            _, context = replay(
                originals, runtime.normalizers, baseline["observations"], query_us, args.device
            )
            exact_arrays(
                context,
                read_arrays(source / "ORIGINAL_CONTEXT.npz"),
                "cached frozen baseline context",
            )
            old_native = native_vector(context, args.device)
            runtime.models = corrected
            destination = args.output / name
            clean = execute(runtime, calls, arrays, query_us, 1.0, args.device, destination / "a1")
            report = {
                "baseline_exact": True,
                "clean_native_seconds": finite_ratio(clean["native"], 1.0),
                "original_clean_native_seconds": finite_ratio(old_native, 1.0),
                "clean_accepted_ids": clean["accepted_ids"],
                "factors": {},
            }
            for speed, label in ((0.5, "a0p5"), (2.0, "a2")):
                warped = execute(
                    runtime, calls, arrays, query_us, speed, args.device, destination / label
                )
                changed = set(warped["selected_ids"]) != set(clean["selected_ids"])
                control = (
                    execute(
                        runtime,
                        calls,
                        arrays,
                        query_us,
                        1.0,
                        args.device,
                        destination / (label + "_matched_clean"),
                        only_ids=set(warped["selected_ids"]),
                    )
                    if changed
                    else clean
                )
                warped_groups = [
                    (call["original_call_id"], [o["observation_id"] for o in call["observations"]])
                    for call in warped["calls"]
                ]
                control_groups = [
                    (call["original_call_id"], [o["observation_id"] for o in call["observations"]])
                    for call in control["calls"]
                ]
                if warped_groups != control_groups:
                    raise ValueError("Corrected matched control changed batch identities/order")
                report["factors"][label] = {
                    "speed": speed,
                    "stage_by_observation": compare_stages(warped, control, speed),
                    "native_ttc_ratio_against_matched_control": finite_ratio(
                        speed * warped["native"], control["native"]
                    ),
                    "original_native_ttc_ratio": complete["reports"][name]["factors"][label][
                        "native_ttc_ratio_against_matched_control"
                    ],
                    "head_comparison_isolating": set(warped["accepted_ids"])
                    == set(control["accepted_ids"]),
                    "retained_ids": warped["selected_ids"],
                    "accepted_ids": warped["accepted_ids"],
                    "control_accepted_ids": control["accepted_ids"],
                    "matched_batch_groups_exact": True,
                }
            atomic_receipt(destination / "COMPLETE.json", report)
            reports[name] = report
            print(f"{name}: canonical PAIR replay complete", flush=True)
        verify_artifacts(Path.cwd(), hashes)
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "reports": reports,
                "targets_read": False,
                "optimizer_updates": 0,
                "elapsed_seconds": time.perf_counter() - start,
                "freeze_sha256": file_sha(args.output / "FREEZE.json"),
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
            {
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "targets_read": False,
                "optimizer_updates": 0,
            },
        )
        raise


if __name__ == "__main__":
    main()
