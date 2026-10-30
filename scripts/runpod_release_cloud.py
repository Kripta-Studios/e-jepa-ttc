"""Future-Pod RGB admission and balanced three-seed queue; no provisioning API."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def relocate_config(config: dict[str, Any], root: Path, recipe: str, seed: int) -> dict[str, Any]:
    """Change paths/run identity only; retain the exact scientific recipe."""
    result = copy.deepcopy(config)
    result["seed"] = seed
    result["experiment_id"] = f"cloud-rgb-{recipe}-d4096-seed{seed}"
    result["data"]["root"] = str(root / "drives/E/eAP_dataset")
    result["model"]["portable_root"] = "output/cloud_portable_monodgp_velocity"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "run"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    project = root / "workspace/phase-ttc-s0-evaluation"
    incoming = root / "incoming"
    receipt = json.loads((incoming / "BUILD_RECEIPT.json").read_text())
    verified = json.loads((root / "VERIFIED_rgb.json").read_text())
    if verified["manifest_sha256"] != receipt["manifest_sha256"]:
        raise ValueError("The transferred RGB dataset has not passed verification")
    os.chdir(project)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    frozen = project / "output/detection_campaign/rgb_screen_v5_seed7_v1"
    sys.path.insert(0, str(frozen / "producer/src"))
    sys.path.insert(0, str(frozen / "producer/scripts"))
    helper_path = frozen / "automatic_queue_v1/scripts/run_detection_campaign_queue.py"
    spec = importlib.util.spec_from_file_location("cloud_frozen_helper", helper_path)
    if spec is None or spec.loader is None:
        raise ValueError("Missing frozen verifier")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    config_root = root / "cloud_configs"
    queue_root = root / "cloud_runs"
    if args.mode == "prepare":
        config_root.mkdir()
        originals = {
            "r0": frozen / "configs/r0.yaml",
            "r2": root / "campaign/autonomous_rgb_queue_v1/configs/r2_seed7.yaml",
        }
        rows = []
        from e_jepa_ttc.detection.campaign_training import (
            validate_campaign_manifest,
            validate_manifest_against_metadata,
        )

        for seed in (7, 13, 23):
            for recipe, original in originals.items():
                config = relocate_config(yaml.safe_load(original.read_text()), root, recipe, seed)
                path = config_root / f"{recipe}_seed{seed}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                loaded = helper.load_campaign_config(path)
                metadata = validate_manifest_against_metadata(
                    validate_campaign_manifest(loaded), loaded
                )
                rows.append(
                    {
                        "name": path.stem,
                        "path": str(path),
                        "config_sha256": loaded.sha256,
                        "file_sha256": helper.sha256_file(path),
                        "source_config_sha256": helper.sha256_file(original),
                        "metadata_status": metadata.get("status", "VALIDATED_WITHOUT_EXCEPTION"),
                    }
                )
        (root / "CLOUD_CPU_ADMISSION.json").write_text(
            json.dumps(
                {
                    "status": "PASS_METADATA_AND_RECIPE",
                    "jobs": rows,
                    "parent_local_results_preserved": True,
                    "cold_cloud_replicates": True,
                    "windows_checkpoint_resume": "PRESERVED_NOT_AUTOMATICALLY_MIGRATED",
                    "claim": (
                        "Fresh matched cloud replications; "
                        "not bitwise hardware parity or final model"
                    ),
                },
                indent=2,
            )
        )
        return
    admission = json.loads((root / "CLOUD_CPU_ADMISSION.json").read_text())
    if admission["status"] != "PASS_METADATA_AND_RECIPE":
        raise ValueError("Run prepare first")
    queue_root.mkdir(exist_ok=True)
    producer = frozen / "producer/scripts/train_detection_campaign.py"
    if args.mode == "smoke":
        path = config_root / "r0_seed7.yaml"
        output = queue_root / "hardware_smoke"
        if output.exists():
            raise FileExistsError("Preserve the previous hardware smoke output")
        helper.resource_preflight(helper.load_campaign_config(path), "0")
        subprocess.run(
            [
                sys.executable,
                str(producer),
                "--config",
                str(path),
                "--output",
                str(output),
                "--stop-after-update",
                "16",
            ],
            check=True,
        )
        report = json.loads((output / "summary.json").read_text())
        if report["final_update"] != 16:
            raise ValueError("Hardware smoke did not complete 16 updates")
        (root / "CLOUD_GPU_ADMISSION.json").write_text(
            json.dumps(
                {
                    "status": "PASS_EXECUTION_ONLY",
                    "summary_sha256": helper.sha256_file(output / "summary.json"),
                    "device": report["device"],
                    "stage_training_seconds": report["stage_training_seconds"],
                    "peak_reserved_bytes": report["peak_reserved_bytes"],
                    "accuracy_claim": False,
                },
                indent=2,
            )
        )
        return
    hardware = json.loads((root / "CLOUD_GPU_ADMISSION.json").read_text())
    if hardware["status"] != "PASS_EXECUTION_ONLY" or hardware[
        "summary_sha256"
    ] != helper.sha256_file(queue_root / "hardware_smoke/summary.json"):
        raise ValueError("Hardware gate missing or changed")
    with (queue_root / "OWNER.json").open("x") as stream:
        json.dump({"pid": os.getpid()}, stream)
    (queue_root / "receipts").mkdir()
    populations = None
    records = {}
    for job in admission["jobs"]:
        if (queue_root / "STOP_AFTER_CURRENT").exists():
            raise InterruptedError("Next job held by user request")
        path = Path(job["path"])
        if helper.sha256_file(path) != job["file_sha256"]:
            raise ValueError("Cloud configuration changed after admission")
        config = helper.load_campaign_config(path)
        helper.resource_preflight(config, "0")
        output = queue_root / job["name"]
        (queue_root / "status.json").write_text(
            json.dumps(
                {"status": "RUNNING", "active_job": job["name"], "completed": records}, indent=2
            )
        )
        helper.run_child(
            name=job["name"],
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
        full = helper.verify_full_evaluations(output, result["summary"])
        if populations is None:
            populations = full
        helper.verify_population_same(populations, full)
        records[job["name"]] = {
            "metric": result["summary"]["best_selection_metric"],
            "summary_sha256": result["summary_sha256"],
        }
    (queue_root / "status.json").write_text(
        json.dumps(
            {
                "status": "RGB_CONFIRMATIONS_COMPLETE",
                "completed": records,
                "next": "Review diagnostics and freeze winner before dependent S2-S6 experiments",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
