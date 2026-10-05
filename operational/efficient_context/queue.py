"""Complete resumable queue with independent storage dependencies and no historic trainers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Campaign, atomic_bytes, atomic_json, digest, read


def dependencies(c: Campaign) -> dict:
    """Check exact named paths; absence never becomes a negative scientific result."""
    config = read(Path(c.launch["source_configuration"]))
    labels = config["expansion"]["0"]
    garl_root = Path(c.local["garl_annotations_candidate"]).parents[1]
    train_files = [garl_root / labels[k]["relative_path"] for k in ("metadata", "labels")]
    native = [
        Path(c.local["garl_code_candidate"]) / "configs/ablation/event_lhr.yaml",
        *train_files,
    ]
    route = read(c.historical / "artifacts/simplex_t/shared_gpu_route_20261004/PROTOCOL.json")
    raw = [Path(v["path"]) for v in route["raw"]]
    native_raw = [
        c.raw / sequence / "events.h5"
        for sequence in sorted(set(config["original_sequences"] + config["expansion_sequences"]))
    ]
    return {
        "observed_utc": datetime.now(UTC).isoformat(),
        "wide_missing": [str(v) for v in train_files if not v.is_file()],
        "E1_missing": [str(v) for v in raw if not v.is_file()],
        "garl_missing": [str(v) for v in native if not v.is_file()],
        "garl_raw_missing": [str(v) for v in native_raw if not v.is_file()],
        "TRAIN_metadata_sha256": labels["metadata_sha256"],
        "TRAIN_labels_sha256": labels["labels_sha256"],
        "global_environment_changed": False,
    }


def call(c: Campaign, key: str, module: str, *args: str) -> int:
    """One foreground worker at a time, durable stdout/stderr and command receipts."""
    logs = c.out / "queue/logs"
    logs.mkdir(parents=True, exist_ok=True)
    import psutil

    own = psutil.Process()
    ancestors = {os.getpid(), *(parent.pid for parent in own.parents())}
    busy = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info["pid"] in ancestors:
            continue
        command_line = " ".join(process.info["cmdline"] or []).lower()
        scientific = any(
            v in command_line
            for v in (
                "operational.efficient_context.run train",
                "operational.efficient_context.garl_train",
                "operational.efficient_context.garl_heads",
                "stage70",
                "stage71",
                "stage72",
                "stage73",
                "stage74",
                "stage75",
                "stage76",
            )
        )
        if scientific and ("python" in command_line) and ("train" in command_line):
            busy.append(process.info["pid"])
    if busy:
        atomic_json(
            c.out / "queue/RESOURCE_PAUSE.json",
            {"status": "PAUSED_RESOURCE", "live_scientific_trainer_pids": busy, "task": key},
        )
        return 3
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = logs / f"{timestamp}_{key}.txt"
    command = [sys.executable, "-m", module, *args, "--protocol", str(c.config_path)]
    atomic_json(
        c.out / "QUEUE_PROGRESS.json",
        {"status": "RUNNING", "task": key, "command": command, "log": str(path)},
    )
    entry = {"task": key, "command": command, "start_utc": datetime.now(UTC).isoformat()}
    print("QUEUE_START", key, flush=True)
    with path.open("xb") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ, PYTHONUTF8="1"),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    entry.update(
        returncode=result.returncode,
        end_utc=datetime.now(UTC).isoformat(),
        log=str(path),
        log_sha256=digest(path),
    )
    log = c.out / "COMMAND_LOG.jsonl"
    atomic_bytes(
        log,
        (log.read_bytes() if log.exists() else b"")
        + (json.dumps(entry, ensure_ascii=False) + "\n").encode(),
    )
    atomic_json(c.out / "QUEUE_PROGRESS.json", entry)
    print("QUEUE_END", key, result.returncode, flush=True)
    return result.returncode


def completed_wide(c: Campaign) -> bool:
    """Reuse the completed family only after checking all nine sealed checkpoint bytes."""
    aggregate = c.out / "WIDE_REPLICATION_RESULTS.json"
    if not aggregate.exists() or read(aggregate).get("status") != "COMPLETE":
        return False
    for seed in (7, 13, 23):
        seal_path = c.out / f"ENDPOINTS_seed{seed}.json"
        result_path = c.out / (
            "H8_WIDE_RESULTS.json" if seed == 7 else f"H8_WIDE_RESULTS_seed{seed}.json"
        )
        if not seal_path.exists() or not result_path.exists():
            return False
        seal = read(seal_path)
        if not seal["all_three_frozen_before_evaluation"] or len(seal["fits"]) != 3:
            return False
        if read(result_path).get("status") != "COMPLETE":
            return False
        for fit in seal["fits"]:
            if fit["updates"] != 2500 or fit["seed"] != seed:
                raise ValueError("completed WIDE endpoint has changed its fixed recipe")
            if digest(Path(fit["checkpoint"])) != fit["checkpoint_sha256"]:
                raise ValueError("completed WIDE checkpoint bytes differ; do not retrain")
    return True


def all_tasks(c: Campaign) -> int:
    """Continue every currently viable branch; no source fallback or truncation."""
    c.freeze()
    qa = c.out / "TEST_RESULTS/final/QA.json"
    if not qa.exists() or read(qa)["status"] != "PASSED":
        raise ValueError("run final QA successfully before new E1/E3 execution")
    state = {"E0": "PENDING", "E1": "PENDING", "E2": "PENDING", "E3": "PENDING"}
    if not (c.out / "EWMA_TRANSPORT_CV_RESULTS.json").exists():
        call(c, "analytical", "operational.efficient_context.analytical_control")
    state["E0"] = (
        "COMPLETE" if (c.out / "EWMA_TRANSPORT_CV_RESULTS.json").exists() else "INCOMPLETE"
    )
    dep = dependencies(c)
    atomic_json(c.out / "DEPENDENCIES.json", dep)
    if not dep["wide_missing"] and completed_wide(c):
        freeze_analysis(c)
        state["E2"] = "COMPLETE"
        atomic_json(
            c.out / "queue/WIDE_REUSED.json",
            {"verified_endpoints": 9, "optimizer_updates": 0, "analysis_reexecuted": False},
        )
    elif not dep["wide_missing"]:
        code = call(c, "wide_seed7", "operational.efficient_context.run", "train", "--resume")
        if code == 0:
            freeze_analysis(c)
            code = call(
                c, "wide_evaluate7", "operational.efficient_context.run", "evaluate", "--seed", "7"
            )
            if code == 0 and read(c.out / "H8_WIDE_RESULTS.json")["replication_authorized"]:
                for seed in (13, 23):
                    code = call(
                        c,
                        f"wide_seed{seed}",
                        "operational.efficient_context.run",
                        "train",
                        "--seed",
                        str(seed),
                        "--resume",
                    )
                    if code:
                        break
                    code = call(
                        c,
                        f"wide_evaluate{seed}",
                        "operational.efficient_context.run",
                        "evaluate",
                        "--seed",
                        str(seed),
                    )
                    if code:
                        break
                if code == 0:
                    code = call(c, "wide_aggregate", "operational.efficient_context.wide_aggregate")
            state["E2"] = "COMPLETE" if code == 0 else "INCOMPLETE_CHECK_LOG"
        else:
            state["E2"] = "INCOMPLETE_CHECK_LOG"
    else:
        state["E2"] = "BLOCKED_DEPENDENCY"
        atomic_json(
            c.out / "H8_WIDE_RESULTS.json",
            {
                "status": "BLOCKED_DEPENDENCY",
                "dependency": dep["wide_missing"],
                "partial_endpoint_evaluated": False,
                "replication_authorized": False,
                "remaining_seed7_updates": 900,
            },
        )
    if not (c.out / "PREPARED_HEAD_RESULTS.json").exists():
        call(c, "prepared_heads", "operational.efficient_context.prepared_heads")
    parity_failures = [
        str(path)
        for path in (c.out / "profile/fragments").glob("*.json")
        if read(path)["parity"]["status"] == "FAILED_INTEGRITY"
    ]
    if parity_failures:
        atomic_json(
            c.out / "INPUT_OUTPUT_PARITY_VALID_DISPATCH_FAILURE.json",
            {
                "status": "FAILED_INTEGRITY",
                "failed_fragments": parity_failures,
                "automatic_retry": False,
                "R0_end_to_end_measured": False,
                "scientific_negative": False,
            },
        )
        safe_failures = [
            str(path)
            for path in (c.out / "profile_reference_dispatch/fragments").glob("*.json")
            if read(path)["parity"]["status"] == "FAILED_INTEGRITY"
        ]
        if safe_failures:
            state["E1"] = "FAILED_INTEGRITY_REQUIRES_INSPECTION"
        elif not dep["E1_missing"]:
            code = call(
                c, "raw_profile_reference_dispatch", "operational.efficient_context.profile_safe"
            )
            state["E1"] = "COMPLETE" if code == 0 else "INCOMPLETE_CHECK_LOG"
        else:
            state["E1"] = "BLOCKED_DEPENDENCY"
    elif not dep["E1_missing"]:
        code = call(c, "raw_profile", "operational.efficient_context.run", "profile")
        state["E1"] = "COMPLETE" if code == 0 else "INCOMPLETE_CHECK_LOG"
    else:
        state["E1"] = "BLOCKED_DEPENDENCY_R0_PREPARED_HEAD_COMPLETE"
        atomic_json(
            c.out / "INPUT_OUTPUT_PARITY.json",
            {
                "status": "RAW_CONTEXTS_BLOCKED_DEPENDENCY",
                "dependencies": dep["E1_missing"],
                "fixture_preprocessing_tests": "PASSED",
                "frozen_producer_dispatch_validation": "PENDING_REAL_INPUTS",
                "R0_end_to_end_measured": False,
                "R1": "NOT_IMPLEMENTED",
                "R2": "HEAD_ONLY_CPU_FP32_MEASURED_SEPARATELY",
                "optimizer_updates": 0,
            },
        )
    if not dep["garl_missing"] and not dep["garl_raw_missing"]:
        cache_qa = c.out / "garl/CACHE_ENGINEERING_QA.json"
        producer_module = (
            "garl_train_cached"
            if cache_qa.exists() and read(cache_qa).get("status") == "PASSED"
            else "garl_train"
        )
        compressed_qa = c.out / "garl/COMPRESSED_CACHE_QA.json"
        if compressed_qa.exists() and read(compressed_qa).get("status") == "PASSED":
            producer_module = "garl_train_compressed"
        parallel_qa = c.out / "garl/PARALLEL_INPUT_QA.json"
        if parallel_qa.exists() and read(parallel_qa).get("status") == "PASSED":
            producer_module = "garl_train_parallel_authorized"
        if (c.out / "garl/QUOTA_SCANNER_FREEZE.json").exists():
            producer_module = "garl_train_scandir"
        stages = [
            ("garl_input_QA", "garl_qa", ()),
            ("garl_microbatch", "native_garl", ("profile",)),
            ("garl_producers", producer_module, ("--resume",)),
            ("garl_heads", "garl_heads", ("all", "--resume")),
            ("garl_runtime", "garl_runtime", ()),
        ]
        for key, module, args in stages:
            if module == "garl_qa" and (c.out / "garl/INPUT_QA.json").exists():
                if read(c.out / "garl/INPUT_QA.json")["status"] == "PASSED":
                    continue
            if module == "native_garl" and (c.out / "garl/MICROBATCH_PROFILE.json").exists():
                if read(c.out / "garl/MICROBATCH_PROFILE.json")["status"] == "ADMITTED":
                    continue
            code = call(c, key, "operational.efficient_context." + module, *args)
            if code:
                state["E3"] = "INCOMPLETE_CHECK_LOG"
                break
        else:
            state["E3"] = "COMPLETE"
    else:
        state["E3"] = "BLOCKED_DEPENDENCY"
        original = read(c.out / "GARL_COMPARISON.json")
        original.update(
            status="BLOCKED_DEPENDENCY",
            current_missing_paths=dep["garl_missing"],
            current_native_raw_missing=dep["garl_raw_missing"],
            scientific_negative=False,
            producer_training_updates=0,
            heads="BLOCKED_BY_NATIVE_PRODUCERS",
        )
        atomic_json(c.out / "GARL_COMPARISON.json", original)
    atomic_json(
        c.out / "QUEUE_STATE.json",
        {
            "branches": state,
            "dependency_snapshot": dep,
            "single_heavy_worker": True,
            "all_tasks_complete": all(v == "COMPLETE" for v in state.values()),
        },
    )
    return 0 if all(v == "COMPLETE" for v in state.values()) else 3


def freeze_analysis(c: Campaign) -> None:
    """Seal new WIDE analysis before its first OLD_DEV forward, without editing training."""
    files = [
        Path(__file__).with_name("analysis.py"),
        Path(__file__).with_name("wide_aggregate.py"),
        ROOT / "src/e_jepa_ttc/simplex_t/practical_comparison.py",
        ROOT / "src/e_jepa_ttc/evaluation/exact_sequence_v10.py",
    ]
    value = {
        "parent_protocol_sha256": digest(c.out / "PROTOCOL.json"),
        "files": [{"path": str(p), "sha256": digest(p)} for p in files],
    }
    path = c.out / "ANALYSIS_PROTOCOL.json"
    if path.exists() and read(path) != value:
        raise ValueError("new WIDE analysis changed after its freeze")
    if not path.exists():
        atomic_json(path, value)


def main() -> int:
    """Actual E0–E3 resumption entrypoint; offline paths are precise dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "status", "package"))
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "4"
    c = Campaign(args.protocol)
    if args.action == "status":
        print(json.dumps(dependencies(c), ensure_ascii=False))
        return 0
    if args.action == "package":
        from .package import package

        package(c)
        return 0
    started = datetime.now(UTC).isoformat()
    code = all_tasks(c)
    log = c.out / "COMMAND_LOG.jsonl"
    receipt = {
        "task": "complete_E0_E3_queue",
        "command": [sys.executable, "-m", "operational.efficient_context.queue", *sys.argv[1:]],
        "start_utc": started,
        "end_utc": datetime.now(UTC).isoformat(),
        "task_returncode": code,
        "package_verification": "reported separately in BUNDLE_VERIFICATION.json",
        "state_sha256": digest(c.out / "QUEUE_STATE.json"),
    }
    atomic_bytes(
        log,
        (log.read_bytes() if log.exists() else b"")
        + (json.dumps(receipt, ensure_ascii=False) + "\n").encode(),
    )
    from .package import package

    package(c)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
