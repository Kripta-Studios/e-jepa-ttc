"""Hand off pending RGB replicas to an admitted evaluation-prefetch producer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def write(path: Path, value: Any) -> None:  # noqa: ANN401 - JSON boundary
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    project = root / "workspace/phase-ttc-s0-evaluation"
    os.chdir(project)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    source = project / "output/cloud_evaluation_prefetch_v1/producer"
    producer = source / "scripts/train_detection_campaign.py"
    sys.path[:0] = [str(source / "src"), str(source / "scripts")]
    frozen = project / "output/detection_campaign/rgb_screen_v5_seed7_v1"
    helper_path = frozen / "automatic_queue_v1/scripts/run_detection_campaign_queue.py"
    spec = importlib.util.spec_from_file_location("evaluation_queue_verifier", helper_path)
    if spec is None or spec.loader is None:
        raise ValueError("Missing frozen verifier")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    from e_jepa_ttc.detection.campaign_training import (
        validate_campaign_manifest,
        validate_manifest_against_metadata,
    )

    receipt_path = root / "evaluation_optimization_v1/deployment_admission.json"
    admission = json.loads(receipt_path.read_text())
    if admission["status"] != "PASS_EXACT_PAYLOAD_AND_TRAINING_SMOKE":
        raise ValueError("Missing evaluation scheduling admission")
    if helper.sha256_file(producer) != admission["runner_sha256"]:
        raise ValueError("Producer changed after admission")
    for relative, expected in admission["modules"].items():
        if helper.sha256_file(source / relative) != expected:
            raise ValueError(f"Admitted module changed: {relative}")
    for path, expected in admission["evidence_sha256"].items():
        if helper.sha256_file(root / path) != expected:
            raise ValueError(f"Admission evidence changed: {path}")
    previous = root / "cloud_runs_input_v1"
    current = json.loads((previous / "status.json").read_text())
    if current["active"] != ["r0_seed7"] or current["completed"] or current["failures"]:
        raise ValueError("Handoff requires only the original R0 seed 7 to be active")
    queue = root / "cloud_runs_eval_v1"
    queue.mkdir()
    config_root = root / "cloud_configs_eval_v1"
    config_root.mkdir()
    write(queue / "OWNER.json", {"pid": os.getpid(), "jobs": 1})
    names = ["r2_seed7", "r0_seed13", "r2_seed13", "r0_seed23", "r2_seed23"]
    jobs = []
    for name in names:
        original = root / "cloud_configs_input_v1" / f"{name}.yaml"
        config = yaml.safe_load(original.read_text())
        config["experiment_id"] += "-eval-prefetch-v1"
        config["model"]["portable_root"] = f"output/cloud_evaluation_prefetch_v1/portable_{name}"
        path = config_root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        loaded = helper.load_campaign_config(path)
        validate_manifest_against_metadata(validate_campaign_manifest(loaded), loaded)
        jobs.append({"name": name, "path": str(path), "sha256": helper.sha256_file(path)})
    write(queue / "admission.json", {"jobs": jobs, "deployment": admission})
    (queue / "receipts").mkdir()
    state: dict[str, Any] = {
        "status": "WAITING_FOR_PARENT_R0",
        "active": [],
        "completed": {},
        "failures": {},
        "parent_run": str(previous / "r0_seed7"),
        "pending": names.copy(),
        "concurrency": 1,
    }
    write(queue / "status.json", state)
    # The old controller checks this before starting any subsequent replica.
    # Its expected InterruptedError is retained in the old log, not suppressed.
    (previous / "STOP_AFTER_CURRENT").write_text("Admitted evaluation-prefetch handoff\n")
    write(
        root / "EVALUATION_HANDOFF.json",
        {
            "status": "WAITING_FOR_PARENT_R0",
            "parent_queue": str(previous),
            "next_queue": str(queue),
            "admission_sha256": helper.sha256_file(receipt_path),
        },
    )
    try:
        parent_summary = previous / "r0_seed7/summary.json"
        while True:
            if parent_summary.exists():
                try:
                    value = json.loads(parent_summary.read_text())
                    if value.get("status") != "COMPLETED":
                        raise ValueError("Parent R0 did not complete its full budget")
                    break
                except json.JSONDecodeError:
                    pass
            old_state = json.loads((previous / "status.json").read_text())
            if old_state.get("failures"):
                raise RuntimeError(f"Parent campaign failed: {old_state['failures']}")
            time.sleep(10)
        parent_config = helper.load_campaign_config(root / "cloud_configs_input_v1/r0_seed7.yaml")
        parent = helper.verify_summary(
            previous / "r0_seed7",
            parent_config,
            expected_status="COMPLETED",
            expected_final_update=8192,
            expected_start_update=0,
            resume_sha256=None,
        )
        population = helper.verify_full_evaluations(previous / "r0_seed7", parent["summary"])
        state["completed"]["r0_seed7"] = {
            "origin": str(previous / "r0_seed7"),
            "summary_sha256": parent["summary_sha256"],
            "metric": parent["summary"]["best_selection_metric"],
        }
        # Let the parent finish verification and observe STOP_AFTER_CURRENT.
        old_owner = json.loads((previous / "OWNER.json").read_text())["pid"]
        deadline = time.monotonic() + 600
        while True:
            try:
                command = Path(f"/proc/{old_owner}/cmdline").read_text()
            except FileNotFoundError:
                break
            if "runpod_input_optimized_queue.py" not in command:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Original queue did not finish its handoff")
            time.sleep(2)
        write(
            root / "ACTIVE_CAMPAIGN.json",
            {
                "queue": str(queue),
                "status_file": str(queue / "status.json"),
                "optimization": "exact_evaluation_prefetch_v1",
                "jobs": 1,
            },
        )
        write(
            root / "EVALUATION_HANDOFF.json",
            {
                "status": "PARENT_VERIFIED_NEW_QUEUE_RUNNING",
                "queue": str(queue),
                "parent_summary_sha256": parent["summary_sha256"],
            },
        )
        for job in jobs:
            name = job["name"]
            path = Path(job["path"])
            if helper.sha256_file(path) != job["sha256"]:
                raise ValueError("Configuration changed after admission")
            if helper.sha256_file(producer) != admission["runner_sha256"]:
                raise ValueError("Producer changed between jobs")
            config = helper.load_campaign_config(path)
            helper.resource_preflight(config, "0")
            state.update(status="RUNNING", active=[name])
            state["pending"].remove(name)
            write(queue / "status.json", state)
            helper.run_child(
                name=name,
                python=Path(sys.executable),
                producer=producer,
                config_path=path,
                config=config,
                output=queue / name,
                queue_output=queue / "receipts",
                gpu="0",
            )
            result = helper.verify_summary(
                queue / name,
                config,
                expected_status="COMPLETED",
                expected_final_update=8192,
                expected_start_update=0,
                resume_sha256=None,
            )
            helper.verify_population_same(
                population, helper.verify_full_evaluations(queue / name, result["summary"])
            )
            state["completed"][name] = {
                "metric": result["summary"]["best_selection_metric"],
                "summary_sha256": result["summary_sha256"],
            }
            state["active"] = []
            write(queue / "status.json", state)
        state["status"] = "RGB_CONFIRMATIONS_COMPLETE"
        write(queue / "status.json", state)
    except BaseException as error:
        state["status"] = "FAILED_NEEDS_REVIEW"
        state["failures"]["controller"] = str(error)
        write(queue / "status.json", state)
        raise


if __name__ == "__main__":
    main()
