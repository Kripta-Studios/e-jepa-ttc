"""Admit and run the exact-input optimization as a separately recorded cloud campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


def write(path: Path, value: Any) -> None:  # noqa: ANN401 - JSON receipt boundary
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--jobs", choices=(1, 2), type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    project = root / "workspace/phase-ttc-s0-evaluation"
    os.chdir(project)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    frozen = project / "output/detection_campaign/rgb_screen_v5_seed7_v1"
    producer_root = project / "output/cloud_input_optimized_v1/producer"
    producer = producer_root / "scripts/train_detection_campaign.py"
    sys.path[:0] = [str(producer_root / "src"), str(producer_root / "scripts")]
    helper_path = frozen / "automatic_queue_v1/scripts/run_detection_campaign_queue.py"
    spec = importlib.util.spec_from_file_location("optimized_queue_verifier", helper_path)
    if spec is None or spec.loader is None:
        raise ValueError("Missing frozen verifier")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    from e_jepa_ttc.detection.campaign_training import (
        validate_campaign_manifest,
        validate_manifest_against_metadata,
    )

    optimization = root / "optimization_v1"
    parity_path = optimization / "resident_gpu_admission.json"
    parity = json.loads(parity_path.read_text())
    if parity["status"] != "PASS_EXACT_INPUT_AND_FORWARD":
        raise ValueError("Real GPU input/forward parity gate has not passed")
    materialization = json.loads((producer_root / "INPUT_OPTIMIZATION_RECEIPT.json").read_text())
    if helper.sha256_file(producer) != materialization["runner_sha256"]:
        raise ValueError("Optimized producer changed after materialization")
    module = producer_root / "src/e_jepa_ttc/detection/resident_input_cache.py"
    if helper.sha256_file(module) != materialization["resident_cache_sha256"]:
        raise ValueError("Resident input cache changed after materialization")
    config_root = root / "cloud_configs_input_v1"
    queue_root = root / "cloud_runs_input_v1"
    admission_path = root / "CLOUD_CPU_ADMISSION_INPUT_V1.json"
    hardware_path = root / "CLOUD_GPU_ADMISSION_INPUT_V1.json"
    if args.mode == "prepare":
        transferred = json.loads((root / "VERIFIED_rgb.json").read_text())
        sealed = json.loads((root / "incoming/BUILD_RECEIPT.json").read_text())
        if transferred["manifest_sha256"] != sealed["manifest_sha256"]:
            raise ValueError("Transferred population changed")
        config_root.mkdir()
        originals = {
            "r0": frozen / "configs/r0.yaml",
            "r2": root / "campaign/autonomous_rgb_queue_v1/configs/r2_seed7.yaml",
        }
        jobs = []
        for seed in (7, 13, 23):
            for recipe, original in originals.items():
                config = yaml.safe_load(original.read_text())
                name = f"{recipe}_seed{seed}"
                config["seed"] = seed
                config["experiment_id"] = f"cloud-rgb-{recipe}-d4096-seed{seed}-input-v1"
                config["data"]["root"] = str(root / "drives/E/eAP_dataset")
                config["model"]["portable_root"] = (
                    f"output/cloud_input_optimized_v1/portable_{name}"
                )
                path = config_root / f"{name}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                loaded = helper.load_campaign_config(path)
                validate_manifest_against_metadata(validate_campaign_manifest(loaded), loaded)
                jobs.append(
                    {
                        "name": name,
                        "path": str(path),
                        "sha256": helper.sha256_file(path),
                        "source_config_sha256": helper.sha256_file(original),
                    }
                )
        write(
            admission_path,
            {
                "status": "PASS",
                "jobs": jobs,
                "cold_replicates": True,
                "runner_sha256": helper.sha256_file(producer),
                "parity_receipt_sha256": helper.sha256_file(parity_path),
                "parent_runs": str(root / "cloud_runs"),
                "input_optimization": materialization,
            },
        )
        return
    admission = json.loads(admission_path.read_text())
    if (
        admission["status"] != "PASS"
        or admission["runner_sha256"] != helper.sha256_file(producer)
        or admission["parity_receipt_sha256"] != helper.sha256_file(parity_path)
    ):
        raise ValueError("Admission changed")
    queue_root.mkdir(exist_ok=True)
    if args.mode == "smoke":
        output = queue_root / "hardware_smoke"
        if output.exists():
            raise FileExistsError(output)
        config = config_root / "r0_seed7.yaml"
        subprocess.run(
            [
                sys.executable,
                str(producer),
                "--config",
                str(config),
                "--output",
                str(output),
                "--stop-after-update",
                "16",
            ],
            check=True,
        )
        result = json.loads((output / "summary.json").read_text())
        if result["final_update"] != 16 or result["status"] != "BUDGET_STOPPED":
            raise ValueError("Optimized smoke did not complete")
        write(
            hardware_path,
            {
                "status": "PASS_EXECUTION_ONLY",
                "summary_sha256": helper.sha256_file(output / "summary.json"),
                "runner_sha256": helper.sha256_file(producer),
            },
        )
        return
    hardware = json.loads(hardware_path.read_text())
    if (
        hardware["status"] != "PASS_EXECUTION_ONLY"
        or hardware["runner_sha256"] != helper.sha256_file(producer)
        or hardware["summary_sha256"]
        != helper.sha256_file(queue_root / "hardware_smoke/summary.json")
    ):
        raise ValueError("Optimized GPU smoke gate is absent or changed")
    if args.jobs == 2:
        concurrency = json.loads((optimization / "concurrency_admission.json").read_text())
        if concurrency["status"] != "PASS" or concurrency["recommended_jobs"] != 2:
            raise ValueError("Two-job concurrency did not improve measured aggregate throughput")
        limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if not limit.exists() or int(limit.read_text()) < 44 * 10**9:
            raise ValueError("Two-job campaign requires at least 44 GB container RAM")
    with (queue_root / "OWNER.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "jobs": args.jobs}, stream)
    (queue_root / "receipts").mkdir()
    lock = threading.Lock()
    failed = threading.Event()
    state: dict[str, Any] = {
        "status": "RUNNING",
        "concurrency": args.jobs,
        "active": [],
        "completed": {},
        "failures": {},
    }

    def run_job(job: dict[str, str]) -> dict[str, Any]:
        name = job["name"]
        if failed.is_set() or (queue_root / "STOP_AFTER_CURRENT").exists():
            raise InterruptedError("Next job held")
        path = Path(job["path"])
        if helper.sha256_file(path) != job["sha256"]:
            raise ValueError("Configuration changed after admission")
        config = helper.load_campaign_config(path)
        helper.resource_preflight(config, "0")
        with lock:
            state["active"].append(name)
            write(queue_root / "status.json", state)
        try:
            output = queue_root / name
            helper.run_child(
                name=name,
                python=Path(sys.executable),
                producer=producer,
                config_path=path,
                config=config,
                output=output,
                queue_output=queue_root / "receipts",
                gpu="0",
            )
            result = helper.verify_summary(
                output,
                config,
                expected_status="COMPLETED",
                expected_final_update=8192,
                expected_start_update=0,
                resume_sha256=None,
            )
            populations = helper.verify_full_evaluations(output, result["summary"])
            with lock:
                state["completed"][name] = {
                    "metric": result["summary"]["best_selection_metric"],
                    "summary_sha256": result["summary_sha256"],
                }
            return populations
        except BaseException as error:
            failed.set()
            with lock:
                state["status"] = "FAILED_NEEDS_REVIEW"
                state["failures"][name] = str(error)
            raise
        finally:
            with lock:
                state["active"].remove(name)
                write(queue_root / "status.json", state)

    population = None
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_job, job) for job in admission["jobs"]]
        try:
            for future in as_completed(futures):
                current = future.result()
                if population is None:
                    population = current
                else:
                    helper.verify_population_same(population, current)
        except BaseException:
            failed.set()
            for future in futures:
                future.cancel()
            raise
    state["status"] = "RGB_CONFIRMATIONS_COMPLETE"
    write(queue_root / "status.json", state)


if __name__ == "__main__":
    main()
